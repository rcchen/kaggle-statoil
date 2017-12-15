'''
Parses a log into a useful format that we can throw into a chart.
'''

import re

def parse_file(path):
  with open(path) as f:
    content = f.readlines()
    content = [x.strip() for x in content]
    i = 0
    while i < len(content):
      epochLine = content[i]
      dataLine = content[i+1]

      epoch = re.search('Epoch (\d+)\/50', epochLine).group(1)
      data = re.search('loss: ((?:\d|\.)*) - acc: ((?:\d|\.)*) - val_loss: ((?:\d|\.)*) - val_acc: ((?:\d|\.)*)', dataLine)

      train_loss = data.group(1)
      train_acc = data.group(2)
      val_loss = data.group(3)
      val_acc = data.group(4)

      i += 2

      outLine = [str(a) for a in [epoch, train_loss, val_loss, train_acc, val_acc]]
      print(",".join(outLine))

parse_file("logs/nn_b1_b2_syn.out")
