import json

'''
Data is in the following format:
{
  id: string,
  band1: float[],
  band2: float[]
}
'''
def read_data(filename):
  return json.load(open(filename))

def main():
  train = read_data("train.json")
  pprint(train[0])

if __name__ == "__main__":
  main()
