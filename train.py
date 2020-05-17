import time

import numpy as np
import pandas as pd;
start = time.perf_counter()
import tensorflow as tf
import argparse
import pickle
import os
from model import Model
from utils import build_dict, build_dataset, batch_iter
import io

import warnings
warnings.filterwarnings("ignore")
# Uncomment next 2 lines to suppress error and Tensorflow info verbosity. Or change logging levels
tf.logging.set_verbosity(tf.logging.FATAL)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def add_arguments(parser):
    parser.add_argument("--num_hidden", type=int, default=150, help="Network size.")
    parser.add_argument("--num_layers", type=int, default=2, help="Network depth.")
    parser.add_argument("--beam_width", type=int, default=50, help="Beam width for beam search decoder.")
    parser.add_argument("--glove", action="store_true", help="Use glove .")
    parser.add_argument("--embedding_size", type=int, default=300, help="Word embedding size.")

    parser.add_argument("--learning_rate", type=float, default=1e-2, help="Anit's Learning rate.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs.")
    parser.add_argument("--keep_prob", type=float, default=0.8, help="Dropout keep prob.")

    parser.add_argument("--toy", action="store_true", help="Use only 500 samples of data")

    parser.add_argument("--with_model", action="store_true", help="Continue from previously saved model")


saveAccuracy=True;
parser = argparse.ArgumentParser()
add_arguments(parser)
args = parser.parse_args()
with open("args.pickle", "wb") as f:
    pickle.dump(args, f)

if not os.path.exists("saved_model"):
    os.mkdir("saved_model")
else:
    if args.with_model:
        old_model_checkpoint_path = open('saved_model/checkpoint', 'r')
        old_model_checkpoint_path = "".join(["saved_model/",old_model_checkpoint_path.read().splitlines()[0].split('"')[1] ])


print("Building dictionary...")
word_dict, reversed_dict, article_max_len, summary_max_len = build_dict("train", args.toy)
print("Loading training dataset...")
train_x, train_y = build_dataset("train", word_dict, article_max_len, summary_max_len, args.toy)


def tensorboard_comp_graph(tf,sess):
    if not os.path.exists('summaries'):
        os.mkdir('summaries')
    if not os.path.exists(os.path.join('summaries','fourth')):
        os.mkdir(os.path.join('summaries','fourth'))

    tf.summary.FileWriter(os.path.join('summaries','fourth'),sess.graph)

# def tensorboard_scalar_graph(tf,sess):
#     if not os.path.exists('GRAPHS'):
#         os.mkdir('GRAPHS')
#     if not os.path.exists(os.path.join('GRAPHS','fifth')):
#         os.mkdir(os.path.join('GRAPHS','fifth'))
#
#     writer=tf.summary.FileWriter(os.path.join('GRAPHS','fifth'),sess.graph)
#     return writer

#with tf.name_scope('performance'):
    # Summaries need to be displayed
    # Whenever you need to record the loss, feed the mean loss to this placeholder
    #tf_loss_ph=tf.placeholder(tf.float32,shape=None,name='loss_summary')
    # Create a scalar summary object for the loss so it can be displayed
    #tf_loss_summary=tf.summary.scalar('loss',tf_loss_ph)


    # # Whenever you need to record the loss, feed the mean test accuracy to this placeholder
    # tf_accuracy_ph=tf.placeholder(tf.float32,shape=None,name='accuracy_summary')
    # # Create a scalar summary object for the accuracy so it can be displayed
    # tf_accuracy_summary=tf.summary.scalar('accuracy',tf_accuracy_ph)
#performance_summaries=tf.summary.merge([tf_loss_summary])
step=0
loss_dict={}
all_loss=[]
with tf.Session() as sess:
    loss_per_epoch=list()
    model = Model(reversed_dict, article_max_len, summary_max_len, args)
    tensorboard_comp_graph(tf,sess)
    print(tf.contrib.slim.model_analyzer.analyze_vars(tf.trainable_variables(),print_info=True))
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(tf.global_variables())
    if 'old_model_checkpoint_path' in globals():
        print("Continuing from previous trained model:" , old_model_checkpoint_path , "...")
        saver.restore(sess, old_model_checkpoint_path )
    batches = batch_iter(train_x, train_y, args.batch_size, args.num_epochs)
    num_batches_per_epoch = (len(train_x) - 1) // args.batch_size + 1
    print("\nIteration starts.")
    print("Number of batches per epoch :", num_batches_per_epoch)

    for batch_x, batch_y in batches:
        batch_x_len = list(map(lambda x: len([y for y in x if y != 0]), batch_x))
        batch_decoder_input = list(map(lambda x: [word_dict["<s>"]] + list(x), batch_y))
        batch_decoder_len = list(map(lambda x: len([y for y in x if y != 0]), batch_decoder_input))
        batch_decoder_output = list(map(lambda x: list(x) + [word_dict["</s>"]], batch_y))

        batch_decoder_input = list(
            map(lambda d: d + (summary_max_len - len(d)) * [word_dict["<padding>"]], batch_decoder_input))
        batch_decoder_output = list(
            map(lambda d: d + (summary_max_len - len(d)) * [word_dict["<padding>"]], batch_decoder_output))

        train_feed_dict = {
            model.batch_size: len(batch_x),
            model.X: batch_x,
            model.X_len: batch_x_len,
            model.decoder_input: batch_decoder_input,
            model.decoder_len: batch_decoder_len,
            model.decoder_target: batch_decoder_output
        }

        #writer_main=tensorboard_scalar_graph(tf,sess)
        #writer_loss=tensorboard_scalar_graph(tf,sess)
        #with tf.device('/gpu:0'):
        _,step,loss=sess.run([model.update,model.global_step,model.loss],
                                                   feed_dict=train_feed_dict)

        loss_per_epoch.append(loss)
        if step % 1000 == 0:
            print("step {0}: loss = {1}".format(step, loss))

        if step % num_batches_per_epoch == 0:
            hours, rem = divmod(time.perf_counter() - start, 3600)
            minutes, seconds = divmod(rem, 60)
            saver.save(sess, "./saved_model/model.ckpt", global_step=step)
            print(" Epoch {0}: Model is saved.".format(step // num_batches_per_epoch),
            "Elapsed: {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds) , "\n")

            avg_loss=np.mean(loss_per_epoch)
            loss_dict={"loss":avg_loss, "epoch":(step // num_batches_per_epoch)}
            all_loss.append(loss_dict)
            # Execute the summaries defined above
            #summ=sess.run(performance_summaries,feed_dict={tf_loss_ph:avg_loss})

            # Write the obtained summaries to the file, so they can be displayed

            #writer_loss.add_summary(summ,(step // num_batches_per_epoch))
            #summary=tf.Summary(value=[tf.Summary.Value(tag="Loss",simple_value=loss)])
            #writer_main.add_summary(summary,global_step=(step // num_batches_per_epoch))

    saver.save(sess,"summaries/fourth//model.ckpt",global_step=step)
    weights=sess.run([model.embeddings],feed_dict=train_feed_dict)[0]

    out_v=io.open('summaries/fourth/vecs.tsv','w',encoding='utf-8')
    out_m=io.open('summaries/fourth/meta.tsv','w',encoding='utf-8')

    for word_num in range(1,model.vocabulary_size):
        word=reversed_dict[word_num]
        embeddings=weights[word_num]
        out_m.write(word + '\n')
        out_v.write('\t'.join(str(x) for x in embeddings) + '\n')


    out_v.close()
    out_m.close()

df=pd.DataFrame(all_loss)
df.to_csv("loss.csv")
