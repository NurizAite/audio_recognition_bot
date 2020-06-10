## Content
* [About audio recognition bot](#basics)
* [How to start](#purpose)
* [Collect your dataset](#assign)
* [Machine learning](#ml)
* [How to run program?](#run)

# <a name="basics"></a> About audio recognition bot

**Audio recognition bot** is a telegram bot that receives voice messages in which you say 5 digits with pauses and tries to guess them.

This bot :

<li> Parse the source code and perform its behavior directly;
<li> Translate source code into some efficient intermediate representation and immediately execute this;
<li> Explicitly execute stored precompiled code made by a compiler which is part of the interpreter system.

#  <a name="purpose"></a> How to start

You can launch bot with a command /start and bot will send you a message in which ask you to send a voice message.
It will answer with 5 numbers if your message is valide and send a message about error otherwise.


# <a name="assign"></a> Collect your dataset

You can collect your dataset with program audio_digits_dataset_bot.py.
This program will save audio in dataset/ogg/, convert to .wav and save in /dataset/wav/.
After that you can split each audio with a program split_by_vad.py on 5 that will contain one digit.
command for splitting
```
for fname in $(ls dataset/wav/* | grep wav); do python3 split_by_vad.py $fname 0.1 0.01 dataset/splitted; done
```
# <a name="ml"></a> Machine learning

**Machine learning** is an application of artificial intelligence (AI) that provides systems the ability to automatically learn and improve from experience without being explicitly programmed. Machine learning focuses on the development of computer programs that can access data and use it learn for themselves.

The process of learning begins with observations or data, such as examples, direct experience, or instruction, in order to look for patterns in data and make better decisions in the future based on the examples that we provide. The primary aim is to allow the computers learn automatically without human intervention or assistance and adjust actions accordingly.

Model is learning on the your dataset whis is located on folder
/dataset
../0
../1
...
../9

Model is learning with the Random Forest Model.

# <a name="run"></a> How to make alive your bot?

First you need to train your model:
```
python3 ml.py
```
Now you have a trained model and can run bot's program:
```
python3 bot.py
```


