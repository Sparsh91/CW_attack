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
import json
import os.path
# from collections import namedtuple
# sys.path.append("DeepSpeech")
# import DeepSpeech
# try:
#     import pydub
# except:
#     print("pydub was not loaded, MP3 compression will not work")
# from tensorflow.python.keras.backend import ctc_label_dense_to_sparse
# from tf_logits import get_logits
# from ds_ctcdecoder import ctc_beam_search_decoder, Scorer
# from deepspeech_training.util.flags import create_flags, FLAGS
# from deepspeech_training.util.config import Config, initialize_globals
import absl


from tensorflow.python.ops import gen_audio_ops as audio_ops
from tensorflow.python.ops import io_ops
# from types import SimpleNamespace
# import pprint
# from absl import logging, flags
# import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow_addons as tfa
import kws_streaming.data.input_data as input_data
from kws_streaming.models import models
from kws_streaming.train import base_parser
import kws_streaming.models.kws_transformer as kws_transformer
from kws_streaming.layers import modes
from kws_streaming.models import model_flags
from kws_streaming.models import utils

from kws_streaming.models import utils

import math

from transformers import AdamWeightDecay

    


class Attack:
    def __init__(self, sess, loss_fn, phrase_length, max_audio_len,
                 learning_rate=10, num_iterations=5000, batch_size=1,
                 mp3=False, l2penalty=float('inf'), restore_path=None, flags=None):
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
        self.phrase_length = phrase_length
        # self.max_audio_len = max_audio_len
        # self.mp3 = mp3

        # Create all the variables necessary
        # they are prefixed with qq_ just so that we know which
        # ones are ours so when we restore the session we don't
        # clobber them.
        



        ######## Updates by kwt

        desired_samples = flags.desired_samples
        self.wav_filename_placeholder_ = tf.placeholder(
          tf.string, [], name='wav_filename')
        wav_loader = io_ops.read_file(self.wav_filename_placeholder_)
        wav_decoder = tf.audio.decode_wav(
          wav_loader, desired_channels=1, desired_samples=desired_samples)

        # Allow the audio sample's volume to be adjusted.
        self.foreground_volume_placeholder_ = tf.placeholder(
          tf.float32, [], name='foreground_volume')
        # signal resampling to generate more training data
        # it will stretch or squeeze input signal proportinally to:
        self.foreground_resampling_placeholder_ = tf.placeholder(tf.float32, [])

        if self.foreground_resampling_placeholder_ != 1.0:
            image = tf.expand_dims(wav_decoder.audio, 0)
            image = tf.expand_dims(image, 2)
            shape = tf.shape(wav_decoder.audio)
            image_resized = tf.image.resize(
                images=image,
                size=(tf.cast((tf.cast(shape[0], tf.float32) *
                               self.foreground_resampling_placeholder_),
                              tf.int32), 1),
                preserve_aspect_ratio=False)
            image_resized_cropped = tf.image.resize_with_crop_or_pad(
                image_resized,
                target_height=desired_samples,
                target_width=1,
            )
            image_resized_cropped = tf.squeeze(image_resized_cropped, axis=[0, 3])
            scaled_foreground = tf.multiply(image_resized_cropped,
                                            self.foreground_volume_placeholder_)
        else:
            scaled_foreground = tf.multiply(wav_decoder.audio,
                                        self.foreground_volume_placeholder_)
        # Shift the sample's start position, and pad any gaps with zeros.
        self.time_shift_padding_placeholder_ = tf.placeholder(
          tf.int32, [2, 2], name='time_shift_padding')
        self.time_shift_offset_placeholder_ = tf.placeholder(
          tf.int32, [2], name='time_shift_offset')
        padded_foreground = tf.pad(
          tensor=scaled_foreground,
          paddings=self.time_shift_padding_placeholder_,
          mode='CONSTANT')
        sliced_foreground = tf.slice(padded_foreground,
                                   self.time_shift_offset_placeholder_,
                                   [desired_samples, -1])
        # Mix in background noise.
        self.background_data_placeholder_ = tf.placeholder(
          tf.float32, [desired_samples, 1], name='background_data')
        self.background_volume_placeholder_ = tf.placeholder(
          tf.float32, [], name='background_volume')
        background_mul = tf.multiply(self.background_data_placeholder_,
                                   self.background_volume_placeholder_)
        background_add = tf.add(background_mul, sliced_foreground)
        
        print(background_add.shape)

        ####### C&W

        self.delta = delta = tf.Variable(np.zeros((batch_size, max_audio_len), dtype=np.float32), name='qq_delta')
        self.mask = mask = tf.Variable(np.zeros((batch_size, max_audio_len), dtype=np.float32), name='qq_mask')
        # self.cwmask = cwmask = tf.Variable(np.zeros((batch_size, phrase_length), dtype=np.float32), name='qq_cwmask')
        self.original = original = tf.Variable(np.zeros((batch_size, max_audio_len), dtype=np.float32), name='qq_original')
        self.lengths = lengths = tf.Variable(np.zeros(batch_size, dtype=np.int32), name='qq_lengths')
        self.importance = tf.Variable(np.zeros((batch_size, phrase_length), dtype=np.float32), name='qq_importance')
        self.target_keyword = tf.Variable(np.zeros((batch_size, phrase_length), dtype=np.int32), name='qq_keyword')
        # self.target_phrase_lengths = tf.Variable(np.zeros((batch_size), dtype=np.int32), name='qq_phrase_lengths')
        self.rescale = tf.Variable(np.zeros((batch_size,1), dtype=np.float32), name='qq_phrase_lengths')

        # Initially we bound the l_infty norm by 2000, increase this
        # constant if it's not big enough of a distortion for your dataset.
        self.apply_delta = tf.clip_by_value(delta, -2000/2**15, 2000/2**15)*self.rescale

        # We set the new input to the model to be the abve delta
        # plus a mask, which allows us to enforce that certain
        # values remain constant 0 for length padding sequences.
        self.new_input = new_input = self.apply_delta*mask + background_add


        ##########



        background_clamp = tf.clip_by_value(new_input, -1.0, 1.0)

        if flags.preprocess == 'raw':
            # background_clamp dims: [time, channels]
            # remove channel dim
            self.output_ = tf.squeeze(background_clamp, axis=1)
            # below options are for backward compatibility with previous
            # version of hotword detection on microcontrollers
            # in this case audio feature extraction is done separately from
            # neural net and user will have to manage it.
        elif flags.preprocess == 'mfcc':
            # Run the spectrogram and MFCC ops to get a 2D audio: Short-time FFTs
            # background_clamp dims: [time, channels]
            spectrogram = audio_ops.audio_spectrogram(
                background_clamp,
                window_size=flags.window_size_samples,
                stride=flags.window_stride_samples,
                magnitude_squared=flags.fft_magnitude_squared)
            # spectrogram: [channels/batch, frames, fft_feature]

            # extract mfcc features from spectrogram by audio_ops.mfcc:
            # 1 Input is spectrogram frames.
            # 2 Weighted spectrogram into bands using a triangular mel filterbank
            # 3 Logarithmic scaling
            # 4 Discrete cosine transform (DCT), return lowest dct_coefficient_count
            mfcc = audio_ops.mfcc(
                spectrogram=spectrogram,
                sample_rate=flags.sample_rate,
                upper_frequency_limit=flags.mel_upper_edge_hertz,
                lower_frequency_limit=flags.mel_lower_edge_hertz,
                filterbank_channel_count=flags.mel_num_bins,
                dct_coefficient_count=flags.dct_num_features)
            # mfcc: [channels/batch, frames, dct_coefficient_count]
            # remove channel dim
            self.output_ = tf.squeeze(mfcc, axis=0)


        ########


        self.loss, self.pred = 1000, np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])



        model = models.MODELS[flags.model_name](flags)
        if flags.distill_teacher_json:
            with open(flags.distill_teacher_json, 'r') as f:
                teacher_flags = json.load(f, object_hook=lambda d: SimpleNamespace(
                **{ k: v for k, v in flags.__dict__.items() if not k in d },
                **d))
                teacher_base = models.MODELS[teacher_flags.model_name](teacher_flags)
                hard_labels = tf.keras.layers.Lambda(lambda logits: tf.one_hot(tf.math.argmax(logits, axis=-1), depth=flags.label_count))
                teacher = tf.keras.models.Sequential([teacher_base, hard_labels])
                teacher_base.trainable = False
                teacher.trainable = False
        else:
            teacher = None
            teacher_flags = None

        # base_model = model

        # logging.info(model.summary())

        # save model summary
        utils.save_model_summary(model, flags.train_dir)

        # save model and data flags
        # with open(os.path.join(flags.train_dir, 'flags.txt'), 'wt') as f:
        #     pprint.pprint(flags, stream=f)

        loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=flags.label_smoothing)
        metrics = ['accuracy']

        if flags.optimizer == 'adam':
            optimizer = tf.keras.optimizers.Adam(epsilon=flags.optimizer_epsilon)
        elif flags.optimizer == 'adamw':
            # Exclude some layers for weight decay
            exclude = ["pos_emb", "class_emb", "layer_normalization", "bias"]
            optimizer = AdamWeightDecay(learning_rate=0.05, weight_decay_rate=flags.l2_weight_decay, exclude_from_weight_decay=exclude)
        else:
            raise ValueError('Unsupported optimizer:%s' % flags.optimizer)

        loss_weights = [ 0.5, 0.5, 0.0 ] if teacher else [ 1. ] # equally weight losses form label and teacher, ignore ensemble output
        model.compile(optimizer=optimizer, loss=loss, loss_weights=loss_weights, metrics=metrics)

        if flags.start_checkpoint:
            model.load_weights(flags.start_checkpoint).expect_partial()
            logging.info('Weights loaded from %s', flags.start_checkpoint)





        
        # Set up the Adam optimizer to perform gradient descent for us
        start_vars = set(x.name for x in tf.global_variables())
        # optimizer = tf.train.AdamOptimizer(learning_rate)
        exclude = ["pos_emb", "class_emb", "layer_normalization", "bias"]
        optimizer = AdamWeightDecay(learning_rate=0.05, weight_decay_rate=flags.l2_weight_decay)

        # grad,var = optimizer.compute_gradients(self.loss, [delta], tape=None)[0]
        # self.train = optimizer.apply_gradients([(tf.sign(grad),var)])
        # print(self.output_, self.train)
        end_vars = tf.global_variables()
        new_vars = [x for x in end_vars if x.name not in start_vars]
        
        sess.run(tf.variables_initializer(new_vars+[delta]))
        print("Initialization done.\n")


    def attack(self, audio, lengths, target, toks, finetune=None):
        print("Start attack..\n")
        sess = self.sess

        # Initialize all of the variables
        sess.run(tf.variables_initializer([self.delta]))
        sess.run(self.original.assign(np.array(audio)))
        sess.run(self.lengths.assign((np.array(lengths)-(2*Config.audio_step_samples/3))//320))
        sess.run(self.mask.assign(np.array([[1 if i < l else 0 for i in range(self.max_audio_len)] for l in lengths])))
        sess.run(self.cwmask.assign(np.array([[1 if i < l else 0 for i in range(self.phrase_length)] for l in (np.array(lengths)-1)//320])))
        sess.run(self.target_phrase_lengths.assign(np.array([len(x) for x in target])))
        sess.run(self.target_phrase.assign(np.array([list(t)+[0]*(self.phrase_length-len(t)) for t in target])))
        c = np.ones((self.batch_size, self.phrase_length))
        sess.run(self.importance.assign(c))
        sess.run(self.rescale.assign(np.ones((self.batch_size,1))))

        # Here we'll keep track of the best solution we've found so far
        final_deltas = [None]*self.batch_size

        if finetune is not None and len(finetune) > 0:
            sess.run(self.delta.assign(finetune-audio))
        
        # We'll make a bunch of iterations of gradient descent here
        #now = time.time()
        MAX = self.num_iterations
        first_hits = np.zeros((self.batch_size,))
        best_hits = np.zeros((self.batch_size,))
        for i in range(MAX):
            # Print out some debug information every 10 iterations.
            if i%10 == 0:
                new, delta, pred, loss = sess.run((self.output_, self.delta, self.pred, self.loss))

                


            if self.mp3:
                new = sess.run(self.new_input)
                mp3ed = convert_mp3(new, lengths)
                feed_dict = {self.new_input: mp3ed}
            else:
                feed_dict = {}
                
            # Actually do the optimization step
            d, l, new_input, _ = sess.run((self.delta, self.loss, self.output_,
                                                           self.train),
                                                          feed_dict)
                    
            # Report progress
            print("%.3f"%np.mean(l), "\t", "\t".join("%.3f"%x for x in l))

            # logits = np.argmax(logits,axis=2).T
            for ii in range(self.batch_size):
                # Every 100 iterations, check if we've succeeded
                # if we have (or if it's the final epoch) then we
                # should record our progress and decrease the
                # rescale constant.
                if (self.loss_fn == "CTC" and i%10 == 0 and out_list[ii][0][1] == "".join([toks[x] for x in target[ii]])) \
                   or (i == MAX-1 and final_deltas[ii] is None):
                    # Get the current constant
                    rescale = sess.run(self.rescale)
                    if rescale[ii]*2000 > np.max(np.abs(d)):
                        # If we're already below the threshold, then
                        # just reduce the threshold to the current
                        # point and save some time.
                        print("It's way over", np.max(np.abs(d[ii]))/2000.0)
                        rescale[ii] = np.max(np.abs(d[ii]))/2000.0

                    # Otherwise reduce it by some constant. The closer
                    # this number is to 1, the better quality the result
                    # will be. The smaller, the quicker we'll converge
                    # on a result but it will be lower quality.
                    rescale[ii] *= .8

                    # Adjust the best solution found so far
                    final_deltas[ii] = new_input[ii]

                    print("Worked i=%d ctcloss=%f bound=%f"%(ii, cl[ii], 2000*rescale[ii][0]))
                    
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
    flags = model_flags.update_flags(FLAGS)
    # initialize_globals()
    # These are the tokens that we're allowed to use.
    # The - token is special and corresponds to the epsilon
    # value in CTC decoding, and can not occur in the phrase.
    toks = " abcdefghijklmnopqrstuvwxyz'-"
    
    with tf.Session() as sess:
        finetune = []
        audios = []
        lengths = []
        names = []
        source_dBs = []
        distortions = []
        high_pertub_bounds = []
        low_pertub_bounds = []

        # if FLAGS.output is None:
        #     assert FLAGS.outprefix is not None
        # else:
        #     assert FLAGS.outprefix is None
        #     assert len(FLAGS.input) == len(FLAGS.output)
        # if FLAGS.finetune is not None and len(FLAGS.finetune):
        #     assert len(FLAGS.input) == len(FLAGS.finetune)
            
        # # Load the inputs that we're given
        # # TODO: [FINDBUG] loading multiple inputs is possible, 
        # #       but there are some weird things going on at the end of every transcription 
        # for i in range(len(FLAGS.input)):
        #     fs, audio = wav.read(FLAGS.input[i])
        #     names.append(FLAGS.input[i])
        #     assert fs == 16000
        #     assert audio.dtype == np.int16
        #     if (audio.shape[-1] == 2):
        #         audio = np.squeeze(audio[:,1])
        #         print(audio.shape)
        #     source_dB = 20 * np.log10(np.max(np.abs(audio)))
        #     print('source dB', source_dB)
        #     source_dBs.append(source_dB)
        #     audios.append(list(audio))
        #     lengths.append(len(audio))

        #     if FLAGS.finetune is not None:
        #         finetune.append(list(wav.read(FLAGS.finetune[i])[1]))   
            
        # maxlen = max(map(len,audios))
        # audios = np.array([x+[0]*(maxlen-len(x)) for x in audios])
        # finetune = np.array([x+[0]*(maxlen-len(x)) for x in finetune])
        
        maxlen = 0
        audios = np.zeros((16000,))
        phrase = ""

        # phrase = FLAGS.target 
        # print("\nAttack phrase: ", phrase) 
        
        attack = Attack(sess, 'CTC', len(phrase), maxlen,
                        batch_size=len(audios), flags=flags)

        # start_time = time.time() 
        # deltas, first_hits, best_hits = attack.attack(audios,
        #                        lengths,
        #                        [[toks.index(x) for x in phrase]]*len(audios),
        #                        toks,
        #                        finetune)
        # runtime = time.time() - start_time

        # print("Finished in {}s.".format(runtime))
        # # And now save it to the desired output
        # if FLAGS.mp3:
        #     convert_mp3(deltas, lengths)
        #     copyfile("/tmp/saved.mp3", FLAGS.output[0])
        #     print("Final distortion", np.max(np.abs(deltas[0][:lengths[0]]-audios[0][:lengths[0]])))
        # else:
        #     for i in range(len(FLAGS.input)):
        #         if FLAGS.output is not None:
        #             path = FLAGS.output[i]
        #         else:
        #             path = FLAGS.outprefix+str(i)+".wav"
        #         wav.write(path, 16000,
        #                   np.array(np.clip(np.round(deltas[i][:lengths[i]]),
        #                                    -2**15, 2**15-1),dtype=np.int16))
                
        #         # Define metrics for evaluation
        #         diff = deltas[i][:lengths[i]]-audios[i][:lengths[i]]
        #         high_pertub_bound = np.max(np.abs(diff))
        #         low_pertub_bound = np.min(np.abs(diff[diff!=0]))
        #         distortion = 20 * np.log10(np.max(np.abs(diff))) - source_dBs[i]
        #         high_pertub_bounds.append(high_pertub_bound)
        #         low_pertub_bounds.append(low_pertub_bound)
        #         distortions.append(distortion)
        #         print("Final noise loudness: ", distortion)

    # Create data_dict to store values for csv file
    data_dict = {
        'filename': names,
        'length' : lengths,
        'attack_runtime': [runtime]*len(names),
        'source_dB': source_dBs,
        'noise_loudness': distortions,
        'high_pertubation_bound' : high_pertub_bounds,
        'low_pertubation_bound' : low_pertub_bounds,
        'first_hit' : first_hits,
        'best_hit' : best_hits
    }     
    # df = pd.DataFrame(data_dict, columns=['filename', 'length', 'attack_runtime', 'source_dB', 'noise_loudness', 'high_pertubation_bound', 'low_pertubation_bound', 'first_hit', 'best_hit'])
    # csv_filename = "tmp/attack-{}.csv".format(FLAGS.lang, time.strftime("%Y%m%d-%H%M%S"))    
    # df.to_csv(csv_filename, index=False, header=True)   
 
                
    
    
if __name__ == "__main__":
  # parser for training/testing data and speach feature flags
    parser = base_parser.base_parser()

    # sub parser for model settings
    subparsers = parser.add_subparsers(dest='model_name', help='NN model name')

    # DNN model settings
    # parser_dnn = subparsers.add_parser('dnn')
    # dnn.model_parameters(parser_dnn)

    # # DNN raw model settings
    # parser_dnn_raw = subparsers.add_parser('dnn_raw')
    # dnn_raw.model_parameters(parser_dnn_raw)

    # # LSTM model settings
    # parser_lstm = subparsers.add_parser('lstm')
    # lstm.model_parameters(parser_lstm)

    # # GRU model settings
    # parser_gru = subparsers.add_parser('gru')
    # gru.model_parameters(parser_gru)

    # # SVDF model settings
    # parser_svdf = subparsers.add_parser('svdf')
    # svdf.model_parameters(parser_svdf)

    # # CNN model settings
    # parser_cnn = subparsers.add_parser('cnn')
    # cnn.model_parameters(parser_cnn)

    # # CRNN model settings
    # parser_crnn = subparsers.add_parser('crnn')
    # crnn.model_parameters(parser_crnn)

    # # ATT MH RNN model settings
    # parser_att_mh_rnn = subparsers.add_parser('att_mh_rnn')
    # att_mh_rnn.model_parameters(parser_att_mh_rnn)

    # # ATT RNN model settings
    # parser_att_rnn = subparsers.add_parser('att_rnn')
    # att_rnn.model_parameters(parser_att_rnn)

    # # DS_CNN model settings
    # parser_ds_cnn = subparsers.add_parser('ds_cnn')
    # ds_cnn.model_parameters(parser_ds_cnn)

    # # TC Resnet model settings
    # parser_tc_resnet = subparsers.add_parser('tc_resnet')
    # tc_resnet.model_parameters(parser_tc_resnet)

    # # Mobilenet model settings
    # parser_mobilenet = subparsers.add_parser('mobilenet')
    # mobilenet.model_parameters(parser_mobilenet)

    # # Mobilenet V2 model settings
    # parser_mobilenet_v2 = subparsers.add_parser('mobilenet_v2')
    # mobilenet_v2.model_parameters(parser_mobilenet_v2)

    # # xception model settings
    # parser_xception = subparsers.add_parser('xception')
    # xception.model_parameters(parser_xception)

    # # inception model settings
    # parser_inception = subparsers.add_parser('inception')
    # inception.model_parameters(parser_inception)

    # # inception resnet model settings
    # parser_inception_resnet = subparsers.add_parser('inception_resnet')
    # inception_resnet.model_parameters(parser_inception_resnet)

    # # svdf resnet model settings
    # parser_svdf_resnet = subparsers.add_parser('svdf_resnet')
    # svdf_resnet.model_parameters(parser_svdf_resnet)

    # # ds_tc_resnet model settings
    # parser_ds_tc_resnet = subparsers.add_parser('ds_tc_resnet')
    # ds_tc_resnet.model_parameters(parser_ds_tc_resnet)

    # kws_transformer settings
    parser_kws_transformer = subparsers.add_parser('kws_transformer')
    kws_transformer.model_parameters(parser_kws_transformer)

    FLAGS, unparsed = parser.parse_known_args()
    if unparsed and tuple(unparsed) != ('--alsologtostderr',):
        raise ValueError('Unknown argument: {}'.format(unparsed))
    # tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

    print('\n\n\n Voh aa gaya \n\n\n')
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)