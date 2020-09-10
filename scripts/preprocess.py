import argparse
from sklearn.model_selection import train_test_split
import json

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--input_file')
arg_parser.add_argument('--output_dir')
arg_parser.add_argument('--src_lang')
arg_parser.add_argument('--tgt_lang')
arg_parser.add_argument('--bert_model')
args = arg_parser.parse_args()

def write_to_file(split,X,Y):
    with open(args.output_dir+'/'+split+'.'+args.src_lang,'w') as src,open(args.output_dir+'/'+split+'.'+args.tgt_lang,'w') as tgt:
        for x,y in zip(X,Y):
            try:
                print(dt.cmd2template(y,verbose=True),file=tgt_template)
                print(y,file = tgt_raw)
                print(x,file=src)
            except Exception as e:
                untemplated.append((x,y))
    return untemplated


def split_data(args):
    print('Splitting data')
    with open(args.input_file,'r') as f:
        json_data=json.load(f)
    x=[]
    y=[]
    for key in json_data.keys():
        invocation = json_data[key]['invocation']
        cmd = json_data[key]['cmd']
        x.append(invocation)
        y.append(cmd)

    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=1)
    x_test,x_valid,y_test,y_valid = train_test_split(x_test,y_test,test_size=0.5,random_state=1)
    
    write_to_file('train',x_train,y_train)
    write_to_file('test',x_test,y_test)
    write_to_file('valid',x_valid,y_valid)

split_data(args)

# tokenize(args,args.src_lang)
# tokenize(args,args.tgt_lang)
