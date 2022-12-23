## attack.py -- generate audio adversarial examples
##
## Copyright (C) 2017, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.

import numpy as np
# import tensorflow as tf
from shutil import copyfile
import scipy.io.wavfile as wav
import struct
# import pandas as pd
import time
import sys
from collections import namedtuple

import json
import os
# import sys
# from absl import logging
# import tensorflow.compat.v1 as tf
import tensorflow as tf
from kws_streaming.models import models
from kws_streaming.layers import modes
from kws_streaming.models import model_flags
from kws_streaming.models import utils
import kws_streaming.models.att_mh_rnn as att_mh_rnn
import kws_streaming.models.att_rnn as att_rnn
import kws_streaming.models.cnn as cnn
import kws_streaming.models.crnn as crnn
import kws_streaming.models.dnn as dnn
import kws_streaming.models.dnn_raw as dnn_raw
import kws_streaming.models.ds_cnn as ds_cnn
import kws_streaming.models.ds_tc_resnet as ds_tc_resnet
import kws_streaming.models.gru as gru
import kws_streaming.models.inception as inception
import kws_streaming.models.inception_resnet as inception_resnet
import kws_streaming.models.lstm as lstm
import kws_streaming.models.mobilenet as mobilenet
import kws_streaming.models.mobilenet_v2 as mobilenet_v2
import kws_streaming.models.svdf as svdf
import kws_streaming.models.svdf_resnet as svdf_resnet
import kws_streaming.models.tc_resnet as tc_resnet
import kws_streaming.models.xception as xception
import kws_streaming.models.kws_transformer as kws_transformer
from kws_streaming.train import base_parser
from kws_streaming.train import train
import kws_streaming.train.test as test

# from transformers import AdamWeightDecay


# import absl.flags

# Define arguments to be parsed
# f = absl.flags
# f.DEFINE_multi_string('input', None, 'Input audio .wav file(s), at 16KHz (separated by spaces)')
# f.DEFINE_multi_string('output', None, 'Path for the adversarial example(s)')
# f.DEFINE_string('outprefix', None, 'Prefix of path for adversarial examples')
# f.DEFINE_string('target', None, 'Target transcription')
# f.DEFINE_multi_string('finetune', None, 'Initial .wav file(s) to use as a starting point')
# f.DEFINE_integer('lr', 100, 'Learning rate for optimization')
# f.DEFINE_integer('iterations', 1000, 'Maximum number of iterations of gradient descent')
# f.DEFINE_float('l2penalty', float('inf'), 'Weight for l2 penalty on loss function')
# f.DEFINE_boolean('mp3', False, 'Generate MP3 compression resistant adversarial examples')
# f.DEFINE_string('restore_path', None, 'Path to the DeepSpeech checkpoint (ending in best_dev-1466475)')
# f.DEFINE_string('lang', "en", 'Language of the input audio (English: en, German: de)')

# # Define which arguments are required
# f.mark_flag_as_required('input')
# f.mark_flag_as_required('target')
# f.mark_flag_as_required('restore_path')
    

def convert_mp3(new, lengths):
    import pydub
    wav.write("/tmp/load.wav", 16000,
              np.array(np.clip(np.round(new[0][:lengths[0]]),
                               -2**15, 2**15-1),dtype=np.int16))
    pydub.AudioSegment.from_wav("/tmp/load.wav").export("/tmp/saved.mp3")
    raw = pydub.AudioSegment.from_mp3("/tmp/saved.mp3")
    mp3ed = np.array([struct.unpack("<h", raw.raw_data[i:i+2])[0] for i in range(0,len(raw.raw_data),2)])[np.newaxis,:lengths[0]]
    return mp3ed
    

class Attack:
    def __init__(self, sess, model, learning_rate=10, num_iterations=5000, batch_size=1,
                 mp3=False, l2penalty=float('inf'), restore_path=None):
        """
        Set up the attack procedure.
        Here we create the TF graph that we're going to use to
        actually generate the adversarial examples.
        """
        print("\nInitializing attack..\n")
        self.sess = sess
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.batch_size = batch_size
        # self.phrase_length = phrase_length
        self.max_audio_len = 16000
        # self.mp3 = mp3
        self.model = model

        # Create all the variables necessary
        # they are prefixed with qq_ just so that we know which
        # ones are ours so when we restore the session we don't
        # clobber them.
        self.delta = delta = tf.Variable(np.zeros((batch_size, 16000), dtype=np.float32), name='qq_delta')
        # self.mask = mask = tf.Variable(np.zeros((batch_size, 16000), dtype=np.float32), name='qq_mask')
        # self.cwmask = cwmask = tf.Variable(np.zeros((batch_size, phrase_length), dtype=np.float32), name='qq_cwmask')
        self.original = original = tf.Variable(np.zeros((batch_size, 16000), dtype=np.float32), name='qq_original')
        # self.lengths = lengths = tf.Variable(np.zeros(batch_size, dtype=np.int32), name='qq_lengths')
        # self.importance = tf.Variable(np.zeros((batch_size, phrase_length), dtype=np.float32), name='qq_importance')
        self.target = tf.Variable(np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.float32), name='qq_target')
        # self.target_phrase_lengths = tf.Variable(np.zeros((batch_size), dtype=np.int32), name='qq_phrase_lengths')
        self.rescale = tf.Variable(np.zeros((batch_size,1), dtype=np.float32), name='qq_rescale')

        # self.loss = tf.Variable()

        # Initially we bound the l_infty norm by 10000000, increase this
        # constant if it's not big enough of a distortion for your dataset.

        with tf.GradientTape(persistent=True) as tape:

            self.apply_delta = tf.clip_by_value(delta, -10000000, 10000000)*self.rescale

            # We set the new input to the model to be the abve delta
            # plus a mask, which allows us to enforce that certain
            # values remain constant 0 for length padding sequences.
            self.new_input = new_input = self.apply_delta + original

            # We add a tiny bit of noise to help make sure that we can
            # clip our values to 16-bit integers and not break things.
            noise = tf.compat.v1.random_normal(new_input.shape,
                                     stddev=2)
            self.pass_in = tf.clip_by_value(new_input+noise, -2**15, 2**15-1)

            # Feed this final value to get the logits.

            # And finally restore the graph to make the classifier
            # actually do something interesting.
            # saver = tf.train.Saver([x for x in tf.global_variables() if 'qq' not in x.name])
            # saver.restore(sess, restore_path)
        
            
            # Set up the Adam optimizer to perform gradient descent for us
            start_vars = set(x.name for x in tf.compat.v1.global_variables())
            self.loss_fn = tf.keras.losses.CategoricalCrossentropy()
            #self.optimizer = tf.keras.optimizers.Adam()
            self.optimizer = tf.keras.optimizers.SGD(learning_rate=1e-6)
            self.model.compile(optimizer=self.optimizer, loss=self.loss_fn, metrics=['accuracy'])



            
            end_vars = tf.compat.v1.global_variables()
            new_vars = [x for x in end_vars if x.name not in start_vars]
            
            sess.run(tf.compat.v1.variables_initializer(new_vars + [delta]))
            tape.watch([delta])
            self.pred = self.model(self.pass_in)
            self.loss = loss = self.loss_fn(self.target, self.pred)
        # print(tf.keras.losses.CategoricalCrossentropy()([[0., 1., 0.]], [[1., 0., 0.]]))
        # print("\n\n\nHere\n\n\n", loss, self.pred, self.target.shape)
            grad = tape.gradient(loss, delta)
            print(grad)
            # self.train = self.optimizer.update_step(zip([grad], [delta]))
            grad,var = self.optimizer.compute_gradients(loss, [delta], tape=tape)[0]
            print('\n\n\nhere\n\n\n', grad, var)
            self.train = self.optimizer.apply_gradients([(tf.sign(grad),var)])
        

        # print(pass_in.shape, self.target.shape)
        

        # Convert logits to probs for CTC decoder using softmax
        # self.probs = tf.squeeze(tf.nn.softmax(self.logits, name='logits'))
        
        # Initialize scorer for CTC decoder
        # if FLAGS.scorer_path:
        #     self.scorer = Scorer(FLAGS.lm_alpha, FLAGS.lm_beta,
        #                     FLAGS.scorer_path, Config.alphabet)
        # else:
        #     self.scorer = None
        print("Initialization done.\n")


    def attack(self, audio, target):
        print("Start attack..\n")
        sess = self.sess

        # Initialize all of the variables
        sess.run(tf.compat.v1.variables_initializer([self.delta]))
        sess.run(self.original.assign(np.array(audio, dtype=np.float32)))
        sess.run(self.target.assign(np.array(target).reshape((self.batch_size, 12))))
        sess.run(self.rescale.assign(np.ones((self.batch_size,1))))

        # Here we'll keep track of the best solution we've found so far
        final_deltas = [None]*self.batch_size

       
        
        # We'll make a bunch of iterations of gradient descent here
        #now = time.time()
        MAX = self.num_iterations
        first_hits = np.zeros((self.batch_size,))
        best_hits = np.zeros((self.batch_size,))
        for i in range(MAX):

            d, pred, loss, _ = sess.run((self.delta, self.pred, self.loss, self.train))
            

            # Print out some debug information every 10 iterations.
            if i%100 == 0:
                print(i, loss)
                print(pred, np.argmax(pred))

                
            # Actually do the optimization step

            for ii in range(self.batch_size):
                # Every 100 iterations, check if we've succeeded
                # if we have (or if it's the final epoch) then we
                # should record our progress and decrease the
                # rescale constant.
                if (i%100 == 0 and np.argmax(pred) == np.argmax(target)) \
                   or (i == MAX-1 and final_deltas[ii] is None):
                    # Get the current constant
                    rescale = sess.run(self.rescale)
                    if rescale[ii]*10000000 > np.max(np.abs(d)):
                        # If we're already below the threshold, then
                        # just reduce the threshold to the current
                        # point and save some time.
                        print("It's way over", np.max(np.abs(d[ii]))/10000000.0)
                        rescale[ii] = np.max(np.abs(d[ii]))/10000000.0

                    # Otherwise reduce it by some constant. The closer
                    # this number is to 1, the better quality the result
                    # will be. The smaller, the quicker we'll converge
                    # on a result but it will be lower quality.
                    rescale[ii] *= .8

                    # Adjust the best solution found so far
                    final_deltas[ii] = new_input[ii]

                    print("Worked i=%d loss=%f bound=%f"%(ii, l[ii], 10000000*rescale[ii][0]))
                    
                    if (first_hits[ii] == 0):
                        print("First hit for audio {} at iteration {}".format(ii, i))
                        first_hits[ii]=i
                    else:
                        best_hits[ii]=i

                    sess.run(self.rescale.assign(rescale))

                    # Just for debugging, save the adversarial example
                    # to /tmp so we can see it if we want
                    wav.write("tmp/adv.wav", 16000,
                              np.array(np.clip(np.round(new_input[ii]),
                                               -2**15, 2**15-1),dtype=np.int16))
        
        return final_deltas, first_hits, best_hits  
    

def main(_):
    # print("\n\n\nMadarchod\n\n\n")
    # print(tf.__version__)
    flags = model_flags.update_flags(FLAGS)

    # tf.config.set_visible_devices([], 'GPU')
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    # initialize_globals()
    # These are the tokens that we're allowed to use.
    # The - token is special and corresponds to the epsilon
    # value in CTC decoding, and can not occur in the phrase.
    # toks = " abcdefghijklmnopqrstuvwxyz'-"
    
    with tf.compat.v1.Session() as sess:
        # finetune = []
        audios = []
        # lengths = []
        # names = []
        source_dBs = []
        distortions = []
        high_pertub_bounds = []
        low_pertub_bounds = []


        model = models.MODELS['cnn'](flags)
        print(flags.train_dir + 'new_weights-1')
        weights_path = os.path.join(flags.train_dir + 'new_weights-1')

        latest = tf.train.latest_checkpoint('cnn_weights')
        checkpoint = tf.train.Checkpoint(model = model)
        checkpoint.restore(latest)

        # model.load_weights(weights_path).expect_partial()
        # model_stream = utils.to_streaming_inference(
        #     model, flags, modes.Modes.STREAM_EXTERNAL_STATE_INFERENCE)




        # if FLAGS.output is None:
        #     assert FLAGS.outprefix is not None
        # else:
        #     assert FLAGS.outprefix is None
        #     assert len(FLAGS.input) == len(FLAGS.output)
        # if FLAGS.finetune is not None and len(FLAGS.finetune):
        #     assert len(FLAGS.input) == len(FLAGS.finetune)
            
        # Load the inputs that we're given
        # TODO: [FINDBUG] loading multiple inputs is possible, 
        #       but there are some weird things going on at the end of every transcription 
        for i in range(1):
            fs, audio = wav.read('data2/yes/ffd2ba2f_nohash_4.wav')
            # names.append('input_audio_name')
            assert fs == 16000
            assert audio.dtype == np.int16
            if (audio.shape[-1] == 2):
                audio = np.squeeze(audio[:,1])
                print(audio.shape)
            source_dB = 20 * np.log10(np.max(np.abs(audio)))
            print('source dB', source_dB)
            source_dBs.append(source_dB)
            audios.append(list(audio))
            # lengths.append(len(audio))

            # if FLAGS.finetune is not None:
            #     finetune.append(list(wav.read(FLAGS.finetune[i])[1]))   
            
        # maxlen = max(map(len,audios))
        audios = np.array(audios)
        target = np.array([4])
        one_hot_target = tf.keras.utils.to_categorical(target, num_classes=flags.label_count)
        # print("\n\n\nHERE\n", one_hot_target, "\n\n\n")
        print(audios.shape, one_hot_target.shape)
        # finetune = np.array([x+[0]*(maxlen-len(x)) for x in finetune])
        
        # phrase = FLAGS.target 
        # print("\nAttack phrase: ", phrase) 
        
        attack = Attack(sess, model,
                        batch_size=len(audios))

        start_time = time.time() 
        deltas, first_hits, best_hits = attack.attack(audios, one_hot_target)
        runtime = time.time() - start_time

        print("Finished in {}s.".format(runtime))
        # And now save it to the desired output
        
        for i in range(1):
            path = "adversarial_audio_1.wav"
            wav.write(path, 16000,
                      np.array(np.clip(np.round(deltas[i][:lengths[i]]),
                                       -2**15, 2**15-1),dtype=np.int16))
            
            # Define metrics for evaluation
            diff = deltas[i][:lengths[i]]-audios[i][:lengths[i]]
            high_pertub_bound = np.max(np.abs(diff))
            low_pertub_bound = np.min(np.abs(diff[diff!=0]))
            distortion = 20 * np.log10(np.max(np.abs(diff))) - source_dBs[i]
            high_pertub_bounds.append(high_pertub_bound)
            low_pertub_bounds.append(low_pertub_bound)
            distortions.append(distortion)
            print("Final noise loudness: ", distortion)

    # Create data_dict to store values for csv file
    # data_dict = {
    #     'filename': names,
    #     'length' : lengths,
    #     'attack_runtime': [runtime]*len(names),
    #     'source_dB': source_dBs,
    #     'noise_loudness': distortions,
    #     'high_pertubation_bound' : high_pertub_bounds,
    #     'low_pertubation_bound' : low_pertub_bounds,
    #     'first_hit' : first_hits,
    #     'best_hit' : best_hits
    # }     
    # df = pd.DataFrame(data_dict, columns=['filename', 'length', 'attack_runtime', 'source_dB', 'noise_loudness', 'high_pertubation_bound', 'low_pertubation_bound', 'first_hit', 'best_hit'])
    # csv_filename = "tmp/attack-{}.csv".format(FLAGS.lang, time.strftime("%Y%m%d-%H%M%S"))    
    # df.to_csv(csv_filename, index=False, header=True)   
 
                
# def run_script():
#     create_flags()
#     absl.app.run(main)
    
    
if __name__ == "__main__":
    parser = base_parser.base_parser()

    # sub parser for model settings
    subparsers = parser.add_subparsers(dest='model_name', help='NN model name')

    # DNN model settings
    parser_dnn = subparsers.add_parser('dnn')
    dnn.model_parameters(parser_dnn)

    # DNN raw model settings
    parser_dnn_raw = subparsers.add_parser('dnn_raw')
    dnn_raw.model_parameters(parser_dnn_raw)

    # LSTM model settings
    parser_lstm = subparsers.add_parser('lstm')
    lstm.model_parameters(parser_lstm)

    # GRU model settings
    parser_gru = subparsers.add_parser('gru')
    gru.model_parameters(parser_gru)

    # SVDF model settings
    parser_svdf = subparsers.add_parser('svdf')
    svdf.model_parameters(parser_svdf)

    # CNN model settings
    parser_cnn = subparsers.add_parser('cnn')
    cnn.model_parameters(parser_cnn)

    # CRNN model settings
    parser_crnn = subparsers.add_parser('crnn')
    crnn.model_parameters(parser_crnn)

    # ATT MH RNN model settings
    parser_att_mh_rnn = subparsers.add_parser('att_mh_rnn')
    att_mh_rnn.model_parameters(parser_att_mh_rnn)

    # ATT RNN model settings
    parser_att_rnn = subparsers.add_parser('att_rnn')
    att_rnn.model_parameters(parser_att_rnn)

    # DS_CNN model settings
    parser_ds_cnn = subparsers.add_parser('ds_cnn')
    ds_cnn.model_parameters(parser_ds_cnn)

    # TC Resnet model settings
    parser_tc_resnet = subparsers.add_parser('tc_resnet')
    tc_resnet.model_parameters(parser_tc_resnet)

    # Mobilenet model settings
    parser_mobilenet = subparsers.add_parser('mobilenet')
    mobilenet.model_parameters(parser_mobilenet)

    # Mobilenet V2 model settings
    parser_mobilenet_v2 = subparsers.add_parser('mobilenet_v2')
    mobilenet_v2.model_parameters(parser_mobilenet_v2)

    # xception model settings
    parser_xception = subparsers.add_parser('xception')
    xception.model_parameters(parser_xception)

    # inception model settings
    parser_inception = subparsers.add_parser('inception')
    inception.model_parameters(parser_inception)

    # inception resnet model settings
    parser_inception_resnet = subparsers.add_parser('inception_resnet')
    inception_resnet.model_parameters(parser_inception_resnet)

    # svdf resnet model settings
    parser_svdf_resnet = subparsers.add_parser('svdf_resnet')
    svdf_resnet.model_parameters(parser_svdf_resnet)

    # ds_tc_resnet model settings
    parser_ds_tc_resnet = subparsers.add_parser('ds_tc_resnet')
    ds_tc_resnet.model_parameters(parser_ds_tc_resnet)

    # kws_transformer settings
    parser_kws_transformer = subparsers.add_parser('kws_transformer')
    kws_transformer.model_parameters(parser_kws_transformer)

    FLAGS, unparsed = parser.parse_known_args()
    if unparsed and tuple(unparsed) != ('--alsologtostderr',):
        raise ValueError('Unknown argument: {}'.format(unparsed))
    tf.compat.v1.app.run(main=main, argv=[sys.argv[0]] + unparsed)