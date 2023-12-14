from scipy.io import loadmat
import pickle
prefix = './data/'
data_file = loadmat(prefix + 'YelpChi.mat')
labels = data_file['label'].flatten()
feat_data = data_file['features'].todense().A
# load the preprocessed adj_lists
with open(prefix + 'yelp_homo_adjlists.pickle', 'rb') as file:
    homo = pickle.load(file)
file.close()
with open(prefix + 'yelp_rur_adjlists.pickle', 'rb') as file:
    relation1 = pickle.load(file)
file.close()
with open(prefix + 'yelp_rtr_adjlists.pickle', 'rb') as file:
    relation2 = pickle.load(file)
file.close()
with open(prefix + 'yelp_rsr_adjlists.pickle', 'rb') as file:
    relation3 = pickle.load(file)
file.close()

print(labels)