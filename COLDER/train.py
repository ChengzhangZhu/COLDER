from tools import generate_train_test_samples
import argparse
import cPickle
from graph import SocialGraph
from tqdm import tqdm
import numpy as np
from COLDER import COLDER
from sklearn.metrics import classification_report

parser = argparse.ArgumentParser(description='COLDER Model Training/Testing')
parser.add_argument('--data_sets', default='/data/qli1/Experiment/Qian/Fraud Detection/Yelp Shibuti Datasets/Data/yelp_Zip_data.csv', help='Specific Data Set',
                    dest='data_sets', type=str)
parser.add_argument('--minT', default=1, help='Set the minimun walk for each node, default is 1',
                    dest='minT', type=int)
parser.add_argument('--maxT', default=32, help='Set the maximun walk for each node, default is 32',
                    dest='maxT', type=int)
parser.add_argument('--max_length', default=5, help='Set the maximun walk length, default is 5',
                    dest='max_length', type=int)
parser.add_argument('--p', default=0.15, help='Set the walk stop probability at each step, default is 0.15',
                    dest='p', type=float)
parser.add_argument('--save_name', default='Zip', help='Set the save name of the generated samples,'
                                                               'default is Zip', dest='save_name', type=str)
parser.add_argument('--generate_data', default='N', help='Data Generation (Y/N)', dest='generate_data', type=str)
parser.add_argument('--training', default='N', help='Train COLDER model (Y/N)', dest='training', type=str)
parser.add_argument('--trn_begin_date', default='2006-01-01', help='The begin date of training data', dest='trn_begin_date', type=str)
parser.add_argument('--trn_end_date', default='2007-01-01', help='The end date of training data', dest='trn_end_date', type=str)
parser.add_argument('--tst_begin_date', default='2007-01-02', help='The begin date of test data', dest='tst_begin_date', type=str)
parser.add_argument('--tst_end_date', default='2008-01-01', help='The end date of test data', dest='tst_end_date', type=str)
parser.add_argument('--cold_start', default='N', help='Only cold start testing data (Y/N)', dest='cold_start', type=str)
parser.add_argument('--epochs', default=1, help='The training epochs', dest='epochs', type=int)
parser.add_argument('--load_model', default='N', help='Load existing COLDER model (Y/N)', dest='load_model', type=str)
parser.add_argument('--testing', default='N', help='Test COLDER model (Y/N)', dest='testing', type=str)
parser.add_argument('--paras', default='1,0.1,0.05', help='The coefficient of training objective', dest='paras', type=str)
parser.add_argument('--lr', default='0.001', help='Learning rate', dest='lr', type=float)

args = parser.parse_args()


def main():
    if args.cold_start == 'Y' or args.cold_start == 'y':
        cold_start = True
    else:
        cold_start = False
    if args.generate_data == 'Y' or args.generate_data == 'y':
        generate_train_test_samples(args.data_sets, args.trn_begin_date, args.trn_end_date,
                                    args.tst_begin_date, args.tst_end_date,
                                    save_name=args.save_name, cold_start=cold_start,
                                    minT=args.minT, maxT=args.maxT, p=args.p, max_length=args.max_length)
    if args.training == 'Y' or args.training == 'y':
        print('Begin Training Process...')
        g = cPickle.load(open('{}_{}_{}_graph.cpkl'.format(args.save_name,args.trn_begin_date,args.trn_end_date),'rb'))
        train_data = cPickle.load(open('{}_{}_{}_train_data.cpkl'.format(args.save_name,args.trn_begin_date,args.trn_end_date),'rb'))
        paras = [float(i) for i in args.paras.split(',')]
        alpha = [paras[0], paras[0], paras[1], paras[1]*5, paras[2], paras[2]]
        colder = COLDER(alpha=alpha, lr=args.lr)
        if args.load_model == 'Y' or args.load_model == 'y':
            colder.load(name='{}_{}_{}_{}_{}'.format(args.save_name,args.save_name,args.trn_begin_date,args.trn_end_date,args.paras))
        colder.fit(train_data, g=g, epoch=args.epochs)
        print('Saving Model...')
        colder.save(name='{}_{}_{}_{}_{}'.format(args.save_name,args.save_name,args.trn_begin_date,args.trn_end_date,args.paras))
        print('Finish!')

    if args.testing == 'Y' or args.testing == 'y':
        print('Begin Testing Process...')
        colder = COLDER()
        colder.load(name='{}_{}_{}_{}_{}'.format(args.save_name,args.save_name,args.trn_begin_date,args.trn_end_date,args.paras))
        test_data = cPickle.load(open('{}_{}_{}_test_data.cpkl'.format(args.save_name,args.tst_begin_date,args.tst_end_date),'rb'))
        g = cPickle.load(open('{}_{}_{}_graph.cpkl'.format(args.save_name, args.trn_begin_date, args.trn_end_date), 'rb'))
        test_data['user'], test_data['item'] = g.name_to_id(test_data['user'], test_data['item'])
        test_u = np.asarray([[i] for i in test_data['user']])
        test_i = np.asarray([[i] for i in test_data['item']])
        test_review = np.asarray(test_data['review'])
        test_rating = np.asarray(test_data['rating'])
        test_label = test_data['label']
        pred_label = list()
        for index in tqdm(range(len(test_label))):
            try:
                pred_label.append(colder.predict(np.asarray([test_u[index]]), np.asarray([test_i[index]]), np.asarray([test_review[index]]), np.asarray([test_rating[index]]))[0])
            except:
                print('item {} does not exist'.format(g.item_reverse[test_data['item'][index]]))
                pred_label.append(0)
        for i,l in enumerate(pred_label):
            if l<0.5:
                pred_label[i] = 0
            else:
                pred_label[i] = 1
        print(classification_report(test_label, pred_label))
        if not cold_start:
            f = open('{}_{}_{}_{}_{}_{}_result.txt'.format(args.save_name, args.trn_begin_date, args.trn_end_date, args.tst_begin_date, args.tst_end_date, args.paras), 'w+')
        else:
            f = open('Cold_{}_{}_{}_{}_{}_{}_result.txt'.format(args.save_name, args.trn_begin_date, args.trn_end_date, args.tst_begin_date, args.tst_end_date, args.paras), 'w+')
        print >> f, classification_report(test_label, pred_label)
        f.close()
        print('Testing Finish!')
    return None

if __name__ == "__main__":
    main()