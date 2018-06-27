from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import NiftiDataset
import os
import VNet
import math
import datetime

# tensorflow app flags
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('data_dir', './data',
    """Directory of stored data.""")
tf.app.flags.DEFINE_integer('batch_size',1,
    """Size of batch""")               
tf.app.flags.DEFINE_integer('patch_size',256,
    """Size of a data patch""")
tf.app.flags.DEFINE_integer('patch_layer',8,
    """Number of layers in data patch""")
tf.app.flags.DEFINE_integer('epochs',999999999,
    """Number of epochs for training""")
tf.app.flags.DEFINE_string('log_dir', './tmp/log',
    """Directory where to write training and testing event logs """)
tf.app.flags.DEFINE_float('init_learning_rate',0.000005,
    """Initial learning rate""")
tf.app.flags.DEFINE_float('decay_factor',0.01,
    """Exponential decay learning rate factor""")
tf.app.flags.DEFINE_integer('decay_steps',100,
    """Number of epoch before applying one learning rate decay""")
tf.app.flags.DEFINE_integer('display_step',10,
    """Display and logging interval (train steps)""")
tf.app.flags.DEFINE_integer('save_interval',1,
    """Checkpoint save interval (epochs)""")
tf.app.flags.DEFINE_string('checkpoint_dir', './tmp/ckpt',
    """Directory where to write checkpoint""")
tf.app.flags.DEFINE_string('model_dir','./tmp/model',
    """Directory to save model""")
tf.app.flags.DEFINE_bool('restore_training',True,
    """Restore training from last checkpoint""")
tf.app.flags.DEFINE_float('drop_ratio',0.01,
    """Probability to drop a cropped area if the label is empty. All empty patches will be dropped for 0 and accept all cropped patches if set to 1""")
tf.app.flags.DEFINE_integer('min_pixel',5,
    """Minimum non-zero pixels in the cropped label""")
tf.app.flags.DEFINE_integer('shuffle_buffer_size',5,
    """Number of next_element_trains used in shuffle buffer""")
# tf.app.flags.DEFINE_float('class_weight',0.15,
#     """The weight used for imbalanced classes data. Currently only apply on binary segmentation class (weight for 0th class, (1-weight) for 1st class)""")

def placeholder_inputs(input_batch_shape, output_batch_shape):
    """Generate placeholder variables to represent the the input tensors.
    These placeholders are used as inputs by the rest of the model building
    code and will be fed from the downloaded ckpt in the .run() loop, below.
    Args:
        patch_shape: The patch_shape will be baked into both placeholders.
    Returns:
        images_placeholder: Images placeholder.
        labels_placeholder: Labels placeholder.
    """
    # Note that the shapes of the placeholders match the shapes of the full
    # image and label tensors, except the first dimension is now batch_size
    # rather than the full size of the train or test ckpt sets.
    # batch_size = -1

    images_placeholder = tf.placeholder(tf.float32, shape=input_batch_shape, name="images_placeholder")
    labels_placeholder = tf.placeholder(tf.int32, shape=output_batch_shape, name="labels_placeholder")   
   
    return images_placeholder, labels_placeholder

def dice_coe(output, target, loss_type='jaccard', axis=(1, 2, 3), smooth=1e-5):

    inse = tf.reduce_sum(output * target, axis=axis)

    if loss_type == 'jaccard':
        l = tf.reduce_sum(output * output, axis=axis)
        r = tf.reduce_sum(target * target, axis=axis)
    elif loss_type == 'sorensen':
        l = tf.reduce_sum(output, axis=axis)
        r = tf.reduce_sum(target, axis=axis)
    else:
        raise Exception("Unknow loss_type")
    ## old axis=[0,1,2,3]
    # dice = 2 * (inse) / (l + r)
    # epsilon = 1e-5
    # dice = tf.clip_by_value(dice, 0, 1.0-epsilon) # if all empty, dice = 1
    ## new haodong
    dice = (tf.constant(2.0) * tf.cast(inse,dtype=tf.float32) + tf.constant(smooth)) / (tf.cast(l + r, dtype=tf.float32) + tf.constant(smooth))
    ##
    dice = tf.reduce_mean(dice)
    return dice

def train():
    """Train the Vnet model"""
    with tf.Graph().as_default():
        global_step = tf.train.get_or_create_global_step()

        # patch_shape(batch_size, height, width, depth, channels)
        input_batch_shape = (FLAGS.batch_size, FLAGS.patch_size, FLAGS.patch_size, FLAGS.patch_layer, 1) 
        output_batch_shape = (FLAGS.batch_size, FLAGS.patch_size, FLAGS.patch_size, FLAGS.patch_layer, 1) 
        
        images_placeholder, labels_placeholder = placeholder_inputs(input_batch_shape,output_batch_shape)

        for batch in range(FLAGS.batch_size):
            images_log = tf.cast(images_placeholder[batch:batch+1,:,:,:,0], dtype=tf.uint8)
            labels_log = tf.cast(tf.scalar_mul(255,labels_placeholder[batch:batch+1,:,:,:,0]), dtype=tf.uint8)

            tf.summary.image("image", tf.transpose(images_log,[3,1,2,0]),max_outputs=FLAGS.patch_layer)
            tf.summary.image("label", tf.transpose(labels_log,[3,1,2,0]),max_outputs=FLAGS.patch_layer)
        '''
        import glob
        imagelist = glob.glob(os.path.join(FLAGS.data_dir,'Volume','*.nii.gz'))
        labellist = [imagename.replace('Volume','Label').replace('volume','label') for imagename in imagelist]
        imagetrainlist = imagelist[:int(len(imagelist)/2)]
        labeltrainlist = labellist[:int(len(imagelist)/2)]
        imagetestlist  = imagelist[int(len(imagelist)/2):]
        labeltestlist  = labellist[int(len(imagelist)/2):]
        
        '''
        # Get images and labels
        train_data_dir = os.path.join(FLAGS.data_dir,'training')
        test_data_dir = os.path.join(FLAGS.data_dir,'testing')
        # support multiple image input, but here only use single channel, label file should be a single file with different classes
        image_filename = 'image_windowed.nii'
        label_filename = 'label.nii'
        # image_filename = 'img.nii.gz'
        # label_filename = 'label.nii.gz'

        # Force input pipepline to CPU:0 to avoid operations sometimes ended up at GPU and resulting a slow down
        with tf.device('/cpu:0'):
            # create transformations to image and labels
            trainTransforms = [
                NiftiDataset.Normalization(),
                NiftiDataset.Resample(0.2),
                NiftiDataset.Padding((FLAGS.patch_size, FLAGS.patch_size, FLAGS.patch_layer)),
                NiftiDataset.RandomCrop((FLAGS.patch_size, FLAGS.patch_size, FLAGS.patch_layer),FLAGS.drop_ratio,FLAGS.min_pixel),
                NiftiDataset.RandomNoise()
                ]

            TrainDataset = NiftiDataset.NiftiDataset(
                data_dir=train_data_dir,
                image_filename=image_filename,
                label_filename=label_filename,
                transforms=trainTransforms,
                train=True
                )
            
            trainDataset = TrainDataset.get_dataset()
            trainDataset = trainDataset.shuffle(buffer_size=5)
            trainDataset = trainDataset.batch(FLAGS.batch_size)

            testTransforms = [
                NiftiDataset.Normalization(),
                NiftiDataset.Resample(0.2),
                NiftiDataset.Padding((FLAGS.patch_size, FLAGS.patch_size, FLAGS.patch_layer)),
                NiftiDataset.RandomCrop((FLAGS.patch_size, FLAGS.patch_size, FLAGS.patch_layer),FLAGS.drop_ratio,FLAGS.min_pixel)
                ]

            TestDataset = NiftiDataset.NiftiDataset(
                data_dir=test_data_dir,
                image_filename=image_filename,
                label_filename=label_filename,
                transforms=testTransforms,
                train=True
            )

            testDataset = TestDataset.get_dataset()
            testDataset = testDataset.shuffle(buffer_size=5)
            testDataset = testDataset.batch(FLAGS.batch_size)
            
        train_iterator = trainDataset.make_initializable_iterator()
        next_element_train = train_iterator.get_next()

        test_iterator = testDataset.make_initializable_iterator()
        next_element_test = test_iterator.get_next()

        # Initialize the model
        with tf.name_scope("vnet"):
            logits = VNet.v_net(images_placeholder,input_channels = input_batch_shape[4], output_channels =2)

        # # apply weight to the logic funciton
        # if FLAGS.class_weight <0 or FLAGS.class_weight>1:
        #     print("Class weight should between 0 and 1")
        #     exit()

        # if logits.shape[4] == 2:
        #     class_weight = tf.constant([FLAGS.class_weight,1.0-FLAGS.class_weight]) 
        #     weighted_logits = tf.multiply(logits,class_weight)
        # else:
        #     weighted_logits = logits

        for batch in range(FLAGS.batch_size):
            logits_log_0 = tf.cast(logits[batch:batch+1,:,:,:,0], dtype=tf.uint8)
            logits_log_1 = tf.cast(logits[batch:batch+1,:,:,:,1], dtype=tf.uint8)
            tf.summary.image("logits_0", tf.transpose(logits_log_0,[3,1,2,0]),max_outputs=FLAGS.patch_layer)
            tf.summary.image("logits_1", tf.transpose(logits_log_1,[3,1,2,0]),max_outputs=FLAGS.patch_layer)

        # # Exponential decay learning rate
        # train_batches_per_epoch = math.ceil(TrainDataset.data_size/FLAGS.batch_size)
        # decay_steps = train_batches_per_epoch*FLAGS.decay_steps

        with tf.name_scope("learning_rate"):
            learning_rate = FLAGS.init_learning_rate
        #     learning_rate = tf.train.exponential_decay(FLAGS.init_learning_rate,
        #         global_step,
        #         decay_steps,
        #         FLAGS.decay_factor,
        #         staircase=True)
        tf.summary.scalar('learning_rate', learning_rate)

        # softmax op for probability layer
        with tf.name_scope("softmax"):
            softmax_op = tf.nn.softmax(logits,name="softmax")

        for batch in range(FLAGS.batch_size):
            softmax_log_0 = tf.cast(tf.scalar_mul(255,softmax_op[batch:batch+1,:,:,:,0]), dtype=tf.uint8)
            softmax_log_1 = tf.cast(tf.scalar_mul(255,softmax_op[batch:batch+1,:,:,:,1]), dtype=tf.uint8)
            tf.summary.image("softmax_0", tf.transpose(softmax_log_0,[3,1,2,0]),max_outputs=FLAGS.patch_layer)
            tf.summary.image("softmax_1", tf.transpose(softmax_log_1,[3,1,2,0]),max_outputs=FLAGS.patch_layer)

        # Op for calculating loss
        with tf.name_scope("cross_entropy"):
            loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits,
                labels=tf.squeeze(labels_placeholder, 
                squeeze_dims=[4])))
            # loss_op = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(
            #     logits=logits,
            #     labels=tf.squeeze(labels_placeholder,squeeze_dims=[4]),
            #     weights=[[]]))
        tf.summary.scalar('loss',loss_op)

        # Argmax Op to generate label from logits
        with tf.name_scope("predicted_label"):
            pred = tf.argmax(logits, axis=4 , name="prediction")

        for batch in range(FLAGS.batch_size):
            pred_log = tf.cast(tf.scalar_mul(255,pred[batch:batch+1,:,:,:]), dtype=tf.uint8)
            tf.summary.image("pred", tf.transpose(pred_log,[3,1,2,0]),max_outputs=FLAGS.patch_layer)

        # Accuracy of model
        with tf.name_scope("accuracy"):
            correct_pred = tf.equal(tf.expand_dims(pred,-1), tf.cast(labels_placeholder,dtype=tf.int64))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

        # Dice Similarity
        with tf.name_scope("dice"):
            sorensen = dice_coe(tf.expand_dims(pred,-1),tf.cast(labels_placeholder,dtype=tf.int64), loss_type='sorensen')
            jaccard = dice_coe(tf.expand_dims(pred,-1),tf.cast(labels_placeholder,dtype=tf.int64), loss_type='jaccard')
            sorensen_loss = 1. - sorensen
            jaccard_loss = 1. - jaccard
        tf.summary.scalar('sorensen', sorensen)
        tf.summary.scalar('jaccard', jaccard)
        tf.summary.scalar('sorensen_loss', sorensen_loss)
        tf.summary.scalar('jaccard_loss',jaccard_loss)

        # Training Op
        with tf.name_scope("training"):
            # optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=FLAGS.init_learning_rate)
            train_op = optimizer.minimize(
                loss=loss_op,
                global_step=global_step)

        # # epoch checkpoint manipulation
        start_epoch = tf.get_variable("start_epoch", shape=[1], initializer= tf.zeros_initializer,dtype=tf.int32)
        start_epoch_inc = start_epoch.assign(start_epoch+1)

        # saver
        summary_op = tf.summary.merge_all()
        checkpoint_prefix = os.path.join(FLAGS.checkpoint_dir ,"checkpoint")
        print("Setting up Saver...")
        saver = tf.train.Saver(keep_checkpoint_every_n_hours=5)

        # training cycle
        with tf.Session() as sess:
            # Initialize all variables
            sess.run(tf.global_variables_initializer())
            print("{}: Start training...".format(datetime.datetime.now()))

            # summary writer for tensorboard
            train_summary_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
            test_summary_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test', sess.graph)

            # restore from checkpoint
            if FLAGS.restore_training:
                # check if checkpoint exists
                if os.path.exists(checkpoint_prefix+"-latest"):
                    print("{}: Last checkpoint found at {}, loading...".format(datetime.datetime.now(),FLAGS.checkpoint_dir))
                    latest_checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_dir,latest_filename="checkpoint-latest")
                    saver.restore(sess, latest_checkpoint_path)
            
            print("{}: Last checkpoint epoch: {}".format(datetime.datetime.now(),start_epoch.eval()[0]))
            print("{}: Last checkpoint global step: {}".format(datetime.datetime.now(),tf.train.global_step(sess, global_step)))

            # loop over epochs
            for epoch in np.arange(start_epoch.eval(), FLAGS.epochs):
                # initialize iterator in each new epoch
                sess.run(train_iterator.initializer)
                sess.run(test_iterator.initializer)
                print("{}: Epoch {} starts".format(datetime.datetime.now(),epoch+1))

                # training phase
                '''
                reader = sitk.ImageFileReader()
                for imagepath, labelpath in zip(imagetrainlist,labeltrainlist):
                    
                    reader.SetFileName(imagepath)
                    return reader.Execute()
                '''

                while True:
                    try:
                        [image, label] = sess.run(next_element_train)

                        image = image[:,:,:,:,np.newaxis]
                        label = label[:,:,:,:,np.newaxis]
                        
                        train, train_loss, train_accuracy, train_jaccard, summary = sess.run(
                            [train_op, loss_op, accuracy, jaccard, summary_op], feed_dict={images_placeholder: image, labels_placeholder: label})
                        print("train loss: %.4f, accuracy: %.4f, jaccard: %.4f" % (train_loss, train_accuracy, train_jaccard))
                        train_summary_writer.add_summary(summary, global_step=tf.train.global_step(sess, global_step))

                    except tf.errors.OutOfRangeError:
                        start_epoch_inc.op.run()
                        # print(start_epoch.eval())
                        # save the model at end of each epoch training
                        print("{}: Saving checkpoint of epoch {} at {}...".format(datetime.datetime.now(),epoch+1,FLAGS.checkpoint_dir))
                        saver.save(sess, checkpoint_prefix, 
                            global_step=tf.train.global_step(sess, global_step),
                            latest_filename="checkpoint-latest")
                        print("{}: Saving checkpoint succeed".format(datetime.datetime.now()))
                        break
                # testing phase
                print("{}: Training of epoch {} finishes, testing start".format(datetime.datetime.now(),epoch+1))
                testJaccard = []
                while True:
                    try:
                        [image, label] = sess.run(next_element_test)

                        image = image[:,:,:,:,np.newaxis]
                        label = label[:,:,:,:,np.newaxis]
                        test_loss, test_accuracy, test_jaccard, summary = sess.run(
                            [loss_op, accuracy, jaccard, summary_op], feed_dict={images_placeholder: image, labels_placeholder: label})
                        print("test loss: %.4f, accuracy: %.4f, jaccard: %.4f" % (test_loss, test_accuracy, test_jaccard))
                        testJaccard.append(test_jaccard)

                        #loss, summary = sess.run([loss_op, summary_op], feed_dict={images_placeholder: image, labels_placeholder: label})
                        test_summary_writer.add_summary(summary, global_step=tf.train.global_step(sess, global_step))

                    except tf.errors.OutOfRangeError:
                        print("mean testing Jaccard: {}".format(np.array(testJaccard).mean()))
                        break

        # close tensorboard summary writer
        train_summary_writer.close()
        test_summary_writer.close()

def main(argv=None):
    if not FLAGS.restore_training:
        # clear log directory
        if tf.gfile.Exists(FLAGS.log_dir):
            tf.gfile.DeleteRecursively(FLAGS.log_dir)
        tf.gfile.MakeDirs(FLAGS.log_dir)

        # clear checkpoint directory
        if tf.gfile.Exists(FLAGS.checkpoint_dir):
            tf.gfile.DeleteRecursively(FLAGS.checkpoint_dir)
        tf.gfile.MakeDirs(FLAGS.checkpoint_dir)

        # # clear model directory
        # if tf.gfile.Exists(FLAGS.model_dir):
        #     tf.gfile.DeleteRecursively(FLGAS.model_dir)
        # tf.gfile.MakeDirs(FLAGS.model_dir)

    train()

if __name__=='__main__':
    tf.app.run()