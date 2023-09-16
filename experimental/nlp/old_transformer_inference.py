import numpy as np
import pandas as pd
import re
import tensorflow as tf
import tensorflow_datasets as tfds

tf.random.set_seed(1)

# set pandas viewing options
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("max_colwidth", None)


# pd.reset_option("max_colwidth")

# the source of our data is: https://github.com/nbertagnolli/counsel-chat

# load pretrained weights:
# import gdown
# gdown.download('https://drive.google.com/uc?export=download&id=1rR0HAOKgs0yGAyZwqeJkX1U3W8234BgR','chatbot_transformer_v4.h5',True);


# @title Run to get our function for calculating "attention"
def scaled_dot_product_attention(query, key, value, mask):
    """Calculate the attention weights. """
    matmul_qk = tf.matmul(query, key, transpose_b=True)

    # scale matmul_qk
    depth = tf.cast(tf.shape(key)[-1], tf.float32)
    logits = matmul_qk / tf.math.sqrt(depth)

    # add the mask to zero out padding tokens
    if mask is not None:
        logits += mask * -1e9

    # softmax is normalized on the last axis (seq_len_k)
    attention_weights = tf.nn.softmax(logits, axis=-1)

    output = tf.matmul(attention_weights, value)

    return output


# @title Run this to create the multi-head attention mechanism
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, name="multi_head_attention"):
        super(MultiHeadAttention, self).__init__(name=name)
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.query_dense = tf.keras.layers.Dense(units=d_model)
        self.key_dense = tf.keras.layers.Dense(units=d_model)
        self.value_dense = tf.keras.layers.Dense(units=d_model)

        self.dense = tf.keras.layers.Dense(units=d_model)

    def split_heads(self, inputs, batch_size):
        inputs = tf.reshape(inputs, shape=(batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(inputs, perm=[0, 2, 1, 3])

    def call(self, inputs):
        query, key, value, mask = (
            inputs["query"],
            inputs["key"],
            inputs["value"],
            inputs["mask"],
        )
        batch_size = tf.shape(query)[0]

        # linear layers
        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)

        # split heads
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        # scaled dot-product attention
        scaled_attention = scaled_dot_product_attention(query, key, value, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        # concatenation of heads
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))

        # final linear layer
        outputs = self.dense(concat_attention)

        return outputs


# #### Masking
#
# Remember how we used padding to add `<pad>` to fill in sentences that were too short? We don't want our model to think that `<pad>` is an actual word (they're only there as placeholders), so we'll use something called a **mask**, which will cover up the `"<pad>"` instances in our sequence so that our model ignores them.


# @title Run this chunk to create the masks
def create_padding_mask(x):
    mask = tf.cast(tf.math.equal(x, 0), tf.float32)
    # (batch_size, 1, 1, sequence length)
    return mask[:, tf.newaxis, tf.newaxis, :]


# print(create_padding_mask(tf.constant([[1, 2, 0, 3, 0], [0, 0, 0, 4, 5]])))


def create_look_ahead_mask(x):
    seq_len = tf.shape(x)[1]
    look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
    padding_mask = create_padding_mask(x)
    return tf.maximum(look_ahead_mask, padding_mask)


# print(create_look_ahead_mask(tf.constant([[1, 2, 0, 4, 5]])))


# @title Run this chunk as well, for more processing

# note: this gives positional information (i.e., where the words are in a sentence
# and includes it in the model) since our model isn't recurrent anymore


class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, position, d_model):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(position, d_model)

    def get_angles(self, position, i, d_model):
        angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        return position * angles

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            position=tf.range(position, dtype=tf.float32)[:, tf.newaxis],
            i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
            d_model=d_model,
        )
        # apply sin to even index in the array
        sines = tf.math.sin(angle_rads[:, 0::2])
        # apply cos to odd index in the array
        cosines = tf.math.cos(angle_rads[:, 1::2])

        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]
        return tf.cast(pos_encoding, tf.float32)

    def call(self, inputs):
        return inputs + self.pos_encoding[:, : tf.shape(inputs)[1], :]


# example of this class:

# sample_pos_encoding = PositionalEncoding(50, 512)

# plt.pcolormesh(sample_pos_encoding.pos_encoding.numpy()[0], cmap='RdBu')
# plt.xlabel('Depth')
# plt.xlim((0, 512))
# plt.ylabel('Position')
# plt.colorbar()
# plt.show()


# ### Encoders
#
# Remember encoders from last notebook? They were used to process the inputs and then fed those processed inputs into the decoder. We feed in our patient questions into the encoder, the encoder figures out "the gist" of the patient's question, and then feeds that into the decoder for it to respond appropriately (in this case, predict the appropriate therapist response).
#
# For our Transformer model, we take all the steps above (Multi-Headed Attention and masking) and combine them into a series of layers, which will make up our encoder. Our input into this encoder is going to be our patient questions, and the output of our encoder (and, subsequently, the input to our decoder) will be a matrix (think of it like an Excel table) that has each word, weighed by "how important" it is in the context of the entire sentence (which gives us a measure of how much the model should "pay attention" to that word).
#
# Run the code chunks below to create our encoders!


# @title Let's create our encoder!
# individual encoder layer
def encoder_layer(units, d_model, num_heads, dropout, name="encoder_layer"):
    inputs = tf.keras.Input(shape=(None, d_model), name="inputs")
    padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")

    attention = MultiHeadAttention(d_model, num_heads, name="attention")(
        {"query": inputs, "key": inputs, "value": inputs, "mask": padding_mask}
    )
    attention = tf.keras.layers.Dropout(rate=dropout)(attention)
    attention = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inputs + attention)

    outputs = tf.keras.layers.Dense(units=units, activation="relu")(attention)
    outputs = tf.keras.layers.Dense(units=d_model)(outputs)
    outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)
    outputs = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention + outputs)

    return tf.keras.Model(inputs=[inputs, padding_mask], outputs=outputs, name=name)


# complete encoder architecture
def encoder(vocab_size, num_layers, units, d_model, num_heads, dropout, name="encoder"):
    inputs = tf.keras.Input(shape=(None,), name="inputs")
    padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")

    embeddings = tf.keras.layers.Embedding(vocab_size, d_model)(inputs)
    embeddings *= tf.math.sqrt(tf.cast(d_model, tf.float32))
    embeddings = PositionalEncoding(vocab_size, d_model)(embeddings)

    outputs = tf.keras.layers.Dropout(rate=dropout)(embeddings)

    for i in range(num_layers):
        outputs = encoder_layer(
            units=units,
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            name="encoder_layer_{}".format(i),
        )([outputs, padding_mask])

    return tf.keras.Model(inputs=[inputs, padding_mask], outputs=outputs, name=name)


# ### Decoders
#
# Now that we've created our encoder, we need to create the decoder as well. The purpose of the decoder is to take what the encoder spits out and "decode" it into something that's useful (in our case, a therapist's response). For Transformers, the Decoder will look very similar to our Encoder since it also uses multi-headed attention.
#
# The end goal of the Decoder is to figure out how the patient's question relates to what the therapist's response was. It takes the output of the Encoder step, which essentially tells it how much "attention" should be placed on each word in the patient's question, and from there learns what a therapist would normally say in that situation.
#
# Run the code chunk below to create our decoders!


# @title Let's create our decoder!
def decoder_layer(units, d_model, num_heads, dropout, name="decoder_layer"):
    inputs = tf.keras.Input(shape=(None, d_model), name="inputs")
    enc_outputs = tf.keras.Input(shape=(None, d_model), name="encoder_outputs")
    look_ahead_mask = tf.keras.Input(shape=(1, None, None), name="look_ahead_mask")
    padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")

    attention1 = MultiHeadAttention(d_model, num_heads, name="attention_1")(
        inputs={
            "query": inputs,
            "key": inputs,
            "value": inputs,
            "mask": look_ahead_mask,
        }
    )
    attention1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention1 + inputs)

    attention2 = MultiHeadAttention(d_model, num_heads, name="attention_2")(
        inputs={
            "query": attention1,
            "key": enc_outputs,
            "value": enc_outputs,
            "mask": padding_mask,
        }
    )
    attention2 = tf.keras.layers.Dropout(rate=dropout)(attention2)
    attention2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(
        attention2 + attention1
    )

    outputs = tf.keras.layers.Dense(units=units, activation="relu")(attention2)
    outputs = tf.keras.layers.Dense(units=d_model)(outputs)
    outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)
    outputs = tf.keras.layers.LayerNormalization(epsilon=1e-6)(outputs + attention2)

    return tf.keras.Model(
        inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask],
        outputs=outputs,
        name=name,
    )


# decoder itself
def decoder(vocab_size, num_layers, units, d_model, num_heads, dropout, name="decoder"):
    inputs = tf.keras.Input(shape=(None,), name="inputs")
    enc_outputs = tf.keras.Input(shape=(None, d_model), name="encoder_outputs")
    look_ahead_mask = tf.keras.Input(shape=(1, None, None), name="look_ahead_mask")
    padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")

    embeddings = tf.keras.layers.Embedding(vocab_size, d_model)(inputs)
    embeddings *= tf.math.sqrt(tf.cast(d_model, tf.float32))
    embeddings = PositionalEncoding(vocab_size, d_model)(embeddings)

    outputs = tf.keras.layers.Dropout(rate=dropout)(embeddings)

    for i in range(num_layers):
        outputs = decoder_layer(
            units=units,
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            name="decoder_layer_{}".format(i),
        )(inputs=[outputs, enc_outputs, look_ahead_mask, padding_mask])

    return tf.keras.Model(
        inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask],
        outputs=outputs,
        name=name,
    )


# ### Putting it all together!
#
# Now that we have our encoder and decoder, let's combine them into one model, our Transformer model!
#
# Run the code chunk below to create our Transformer model!


# @title Let's create our Transformer!
def transformer(
    vocab_size, num_layers, units, d_model, num_heads, dropout, name="transformer"
):
    inputs = tf.keras.Input(shape=(None,), name="inputs")
    dec_inputs = tf.keras.Input(shape=(None,), name="dec_inputs")

    enc_padding_mask = tf.keras.layers.Lambda(
        create_padding_mask, output_shape=(1, 1, None), name="enc_padding_mask"
    )(inputs)
    # mask the future tokens for decoder inputs at the 1st attention block
    look_ahead_mask = tf.keras.layers.Lambda(
        create_look_ahead_mask, output_shape=(1, None, None), name="look_ahead_mask"
    )(dec_inputs)
    # mask the encoder outputs for the 2nd attention block
    dec_padding_mask = tf.keras.layers.Lambda(
        create_padding_mask, output_shape=(1, 1, None), name="dec_padding_mask"
    )(inputs)

    enc_outputs = encoder(
        vocab_size=vocab_size,
        num_layers=num_layers,
        units=units,
        d_model=d_model,
        num_heads=num_heads,
        dropout=dropout,
    )(inputs=[inputs, enc_padding_mask])

    dec_outputs = decoder(
        vocab_size=vocab_size,
        num_layers=num_layers,
        units=units,
        d_model=d_model,
        num_heads=num_heads,
        dropout=dropout,
    )(inputs=[dec_inputs, enc_outputs, look_ahead_mask, dec_padding_mask])

    outputs = tf.keras.layers.Dense(units=vocab_size, name="outputs")(dec_outputs)

    return tf.keras.Model(inputs=[inputs, dec_inputs], outputs=outputs, name=name)


# Awesome! Now that we've finished creating our Transformer model, let's use it to create a start of the art chatbot!

# ## Preparing our dataset


# @title Run this cell to download our dataset

# load in our data with this code chunk:
chat_data = pd.read_csv(
    "https://raw.githubusercontent.com/nbertagnolli/counsel-chat/master/data/20200325_counsel_chat.csv"
)

# We will use the same dataset as last time. As a recap, our dataset contains questions and answers taken from conversations between patients and licensed mental health professionals.
#
# **Disclaimer**: Once again, this dataset was made freely available and all data was provided consensually, and in anonymized form. Remember, when working with sensitive data such as medical data, you should *always* get permission first!

# ### Exploring our data
#
# As usual, let's begin by exploring our data. Our data is in a pandas dataframe named `chat_data`.
#
# **Exercise**: Print out the first 5 rows in the dataset.


chat_data.head()

# **Question**: What does each row in our dataset represent?

# Now, let's set our `X` and `y` variables to be, respectively, the input and output for our chatbot.
#
# **Question**: What should we set `X` to be? What should we set `y` to be?


X = chat_data["questionText"]
y = chat_data["answerText"]


# ### Cleaning our data
#
# Before we can begin to use our data, we must preprocess it to clean up any data which we don't want to see. Run the following cell to perform some initial cleaning of our data.
#
#


def preprocess_text(phrase):
    phrase = re.sub(r"\xa0", "", phrase)  # removes "\xa0"
    phrase = re.sub(r"\n", "", phrase)  # removes "\n"
    phrase = re.sub("[.]{1,}", "", phrase)  # removes duplicate "."s
    phrase = re.sub("[ ]{1,}", " ", phrase)  # removes duplicate spaces
    return phrase


# run cleaning function
X = X.apply(preprocess_text)
y = y.apply(preprocess_text)

# ### Splitting up our questions and answers
#
# There's a little more preprocessing we need to do however! We want to keep our phrases relatively short; however, some of the questions and answers in our dataset are several sentences long.
#
# To solve this problem, we'll split up each question and each answer into their constituent sentences. We'll then pair the first sentence of the question with the first sentence of the answer, the second sentence of the question with the second sentence of the answer, and so on until we can't form any more pairs.
#
# For example, suppose we have the following question-answer pair:
# > **Q**: "I am not feeling well today. I feel sad."
#
# > **A**: "Tell me more about how you feel. What have you been up to today?"
#
# First, we would split up the question into its constituent sentences, resulting in `["I am not feeling well today.", "I feel sad."]`. Similarly, we would split up the answer into its constituent sentences, resulting in `["Tell me more about how you feel.", "What have you been up to today?"]`. Finally, we would pair each sentence of the question with its corresponding sentence of the answer, ultimately resulting in two separate question-answer pairs:
# > **Q**: "I am not feeling well today."
#
# > **A**: "Tell me more about how you feel."
#
# and
#
# > **Q**: "I feel sad."
#
# > **A**: "What have you been up to today?"


# run this code chunk, to store all of our question/answer pairs
question_answer_pairs = []

# loop through each combination of question + answer
for (question, answer) in zip(X, y):

    # clean up text inputs

    # example:
    # question = "I am not feeling well today. I feel sad."
    # answer = "Tell me more about how you feel. What have you been up to today?"

    question = preprocess_text(question)
    answer = preprocess_text(answer)

    # split by .
    # example
    # question_arr = ["I am not feeling well today", "I feel sad"]
    # answer_arr = ["Tell me more about how you feel", "What have you been up to?"]
    question_arr = question.split(".")
    answer_arr = answer.split(".")

    # get the maximum length, which will be the shorter of the two
    max_sentences = min(len(question_arr), len(answer_arr))

    # for each combination of question + answer, pair them up
    for i in range(max_sentences):
        # set up Q/A pair
        q_a_pair = []

        # append question, answer to pair (e.g,. first sentence of question + first sentence of answer, etc.)
        q_a_pair.append(question_arr[i])
        q_a_pair.append(answer_arr[i])

        # append to question_answer_pairs
        question_answer_pairs.append(q_a_pair)

# ### Tokenizing and padding our data
#
# The next preprocessing steps that we need to implement are going to be tokenization and padding (which we reviewed in our last notebook). If you recall, tokenization is the process of turning a sentence into an array of the individual words (aka tokens), while padding is a way to add "filler" to make short sentences the same length as long sentences. Now that we've seen how tokenization and padding work, let's actually implement that on our dataset.


# @title Run this cell to tokenize our data

# Build tokenizer using tfds for both questions and answers
tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
    [arr[0] + arr[1] for arr in question_answer_pairs], target_vocab_size=2 ** 13
)

# Define start and end token to indicate the start and end of a sentence
START_TOKEN, END_TOKEN = [tokenizer.vocab_size], [tokenizer.vocab_size + 1]

# Vocabulary size plus start and end token
VOCAB_SIZE = tokenizer.vocab_size + 2

# @title Run this cell to define our tokenization and padding functions!

# maximum sentence length
MAX_LENGTH = 100  # chosen arbitrarily


# tokenize, filter, pad sentences
def tokenize_and_filter(inputs, outputs):
    """
    Tokenize, filter, and pad our inputs and outputs
    """

    # store results
    tokenized_inputs, tokenized_outputs = [], []

    # loop through inputs, outputs
    for (sentence1, sentence2) in zip(inputs, outputs):

        # tokenize sentence
        sentence1 = START_TOKEN + tokenizer.encode(sentence1) + END_TOKEN
        sentence2 = START_TOKEN + tokenizer.encode(sentence2) + END_TOKEN

        # check tokenized sentence max length
        if len(sentence1) <= MAX_LENGTH and len(sentence2) <= MAX_LENGTH:
            tokenized_inputs.append(sentence1)
            tokenized_outputs.append(sentence2)

    # pad tokenized sentences
    tokenized_inputs = tf.keras.preprocessing.sequence.pad_sequences(
        tokenized_inputs, maxlen=MAX_LENGTH, padding="post"
    )
    tokenized_outputs = tf.keras.preprocessing.sequence.pad_sequences(
        tokenized_outputs, maxlen=MAX_LENGTH, padding="post"
    )

    return tokenized_inputs, tokenized_outputs


# @title Run this cell to run tokenization and padding on our dataset
# get questions, answers
questions, answers = tokenize_and_filter(
    [arr[0] for arr in question_answer_pairs], [arr[1] for arr in question_answer_pairs]
)

print("Vocab size: {}".format(VOCAB_SIZE))
print("Number of samples: {}".format(len(questions)))

BATCH_SIZE = 64
BUFFER_SIZE = 20000

# decoder inputs use the previous target as input
# remove START_TOKEN from targets
dataset = tf.data.Dataset.from_tensor_slices(
    (
        {"inputs": questions, "dec_inputs": answers[:, :-1]},
        {"outputs": answers[:, 1:]},
    )
)

dataset = dataset.cache()
dataset = dataset.shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE)
dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

print(dataset)

# ## Training our Transformer
#
# Now that we've done all the hard work of setting up and building our model, let's actually train it!

# ### Initializing our model
#
# First, we must initialize our Transformer object.
#
# **Exercise**: Use our Transformer model above to create a Transformer object with these six parameters:
# 1. vocab_size = `VOCAB_SIZE` (we've defined this variable earlier, so you can use the variable name as is)
# 2. num_layers = 2
# 3. units = 512
# 4. d_model = 256
# 5. num_heads = 8
# 6. dropout = 0.1
#


tf.keras.backend.clear_session()

D_MODEL = 256
model = transformer(VOCAB_SIZE, 2, 512, D_MODEL, 8, 0.1)


# @title Run this cell to initiallize our loss function and learning rate


def loss_function(y_true, y_pred):
    y_true = tf.reshape(y_true, shape=(-1, MAX_LENGTH - 1))

    loss = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction="none"
    )(y_true, y_pred)

    mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
    loss = tf.multiply(loss, mask)

    return tf.reduce_mean(loss)


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


# example of how we can adjust learning rate
# sample_learning_rate = CustomSchedule(d_model=128)

# plt.plot(sample_learning_rate(tf.range(200000, dtype=tf.float32)))
# plt.ylabel("Learning Rate")
# plt.xlabel("Train Step")


# @title Run this cell to compile our model
learning_rate = CustomSchedule(D_MODEL)

optimizer = tf.keras.optimizers.Adam(
    learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9
)


def accuracy(y_true, y_pred):
    # ensure labels have shape (batch_size, MAX_LENGTH - 1)
    y_true = tf.reshape(y_true, shape=(-1, MAX_LENGTH - 1))
    return tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred)


model.compile(optimizer=optimizer, loss=loss_function, metrics=[accuracy])

# ### Fitting our data
#
# **Exercise**: We will now train the model we built on Colab's free GPU! You should train your model for 20 epochs.
#
# **Note: BUT, once it starts working, stop the code chunk!** You'll know that it works when you see a progress bar starting with `Epoch 1/20`.
#
# Why are we ending early? Our model isn't even close to done training; however, actually training our model would take *waaaaay* too long for our class. For reference, if we were to train the GPT-3 transformer-based model on the most powerful GPU in the world, then [it would still take *355 years* to train](https://lambdalabs.com/blog/demystifying-gpt-3/#1)! Quite a long class period!


# model.fit(dataset, epochs=20)


# So... do we just end here? Well, we just so happen to have a version of this model that was trained on this same data. So we'll evaluate that instead!


# model.save_weights("./model_saves/chatbot_transformer_v4.0.h5")


# @title Run this cell to import our pretrained model

model.load_weights("./model_saves/chatbot_transformer_v4.0.h5")


# ### Evaluating our model
#
# Now that we have a working model, let's start playing around with it and see how it does!


# @title Run this code chunk to get the functions that we'll need for this section!
def evaluate(sentence):
    sentence = preprocess_text(sentence)

    sentence = tf.expand_dims(
        START_TOKEN + tokenizer.encode(sentence) + END_TOKEN, axis=0
    )

    output = tf.expand_dims(START_TOKEN, 0)

    for i in range(MAX_LENGTH):
        predictions = model(inputs=[sentence, output], training=False)

        # select the last word from the seq_len dimension
        predictions = predictions[:, -1:, :]
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        # return the result if the predicted_id is equal to the end token
        if tf.equal(predicted_id, END_TOKEN[0]):
            break

        # concatenated the predicted_id to the output which is given to the decoder
        # as its input.
        output = tf.concat([output, predicted_id], axis=-1)

    return tf.squeeze(output, axis=0)


def predict(sentence):
    prediction = evaluate(sentence)

    predicted_sentence = tokenizer.decode(
        [i for i in prediction if i < tokenizer.vocab_size]
    )

    #    print("Input: {}".format(sentence))
    #    print("Output: {}".format(predicted_sentence))

    return predicted_sentence


# ### Testing the Transformer's performance

# Let's tell our program something and see what it says


# @title Type in a phrase and let's see what our mental health chatbot thinks!

# sentence = input("Input Sentence: ")
# print("--------------------")
# # output = predict(sentence)
# _ = predict(sentence)
# print("--------------------")
