from datetime import datetime
from typing import List
import warnings
warnings.filterwarnings("ignore", message="Could not import")

import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt

from synthesized.common.encodings import FeedForwardDSSMEncoding
from synthesized.common.generative import VAEOld
from synthesized.common import ValueFactory


df = pd.DataFrame(dict(
    temperature = np.sin(np.linspace(0,10,256, dtype=np.float32)+np.random.uniform(-0.1, 0.1, size=(256,))),
    temperature2 = -np.sin(np.linspace(0,10,256, dtype=np.float32)+np.random.uniform(-0.1, 0.1, size=(256,)))
))

vf = ValueFactory(df, capacity=4)

df_train = vf.preprocess(df)

vae = VAEOld(
    name='vae', values=vf.get_values(), conditions=[], distribution='normal', latent_size=8,
    network='dense', capacity=8, learning_rate=tf.constant(3e-3, dtype=tf.float32), decay_steps=None, decay_rate=None,
    initial_boost=0, beta=1.0, weight_decay=1e-3, optimizer='adam', clip_gradients=1.0,
    batchnorm=True, activation='none'
)

data = vf.get_data_feed_dict(df_train)

global_step = tf.Variable(initial_value=0, trainable=False, dtype=tf.int64)
tf.summary.experimental.set_step(global_step)
stamp = datetime.now().strftime("%Y%m%d-%H%M%S")

# edit this
# logdir = f"/Users/simonhkswan/PycharmProjects/synthesized/logs/ffdss/{stamp}"
logdir =

writer = tf.summary.create_file_writer(logdir)
writer.set_as_default()
tf.summary.trace_on(graph=True, profiler=False)

vae.build()
vae.learn(xs=data)
tf.summary.trace_export(name="Learn", step=global_step)
tf.summary.trace_off()


# @tf.function
def synthesize() -> List[tf.Tensor]:

    init_z = tf.random.normal(shape=[1,8])
    init_h = vae.encoding.init_h()

    z = vae.encoding.sample(inputs=init_h, state=init_z)
    # init_y = vae.decoder(z)
    init_y = vae.linear_output(z)
    ys = tf.split(
        value=init_y, num_or_size_splits=[value.learned_output_size() for value in vae.values],
        axis=1
    )

    outputs = [[] for value in vae.values]
    # Output tensors per value
    for i, (value, y) in enumerate(zip(vae.values, ys)):
        outputs[i].append(value.output_tensors(y=y)[0])

    for index in range(1, 256):
        x = tf.concat(values=[
            value.unify_inputs(xs=[outputs[i][index - 1]]) for i, value in enumerate(vae.values)
        ], axis=1)

        x = vae.linear_input(x)
        # x = vae.encoder(x)
        z = vae.encoding.sample(inputs=x, state=z)
        # print(z)
        # loc_y = vae.decoder(z)
        loc_y = vae.linear_output(z)
        # print(loc_y)

        # Split output tensors per value
        ys = tf.split(
            value=loc_y, num_or_size_splits=[value.learned_output_size() for value in vae.values],
            axis=1
        )

        # Output tensors per value
        for j, (value, loop_y) in enumerate(zip(vae.values, ys)):
            outputs[j].append(value.output_tensors(y=loop_y)[0])

    outputs = {v.name: tf.concat(o, axis=0) for v, o in zip(vae.values, outputs)}

    return outputs


for _ in range(100):
    for i in range(100):
        vae.learn(xs=data)
        global_step.assign_add(1)
    syn = pd.DataFrame(synthesize())
    syn = vf.postprocess(syn)
    fig = plt.figure(figsize=(16,4))
    ax=fig.gca()
    sns.lineplot(y=syn['temperature'], x=syn.index, axes=ax)
    sns.lineplot(y=syn['temperature2'], x=syn.index, axes=ax)
    sns.lineplot(y=df['temperature'], x=df.index, axes=ax)
    sns.lineplot(y=df['temperature2'], x=df.index, axes=ax)
    plt.suptitle(f"Step_{_}")
    # plt.savefig('/Users/simonhkswan/PycharmProjects/synthesized/logs/ffdss/synthesized.png', dpi=100)
    plt.show()