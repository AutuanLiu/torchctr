from torchctr.datasets import get_movielens, read_data

# step 1: download dataset
root = get_movielens('datasets', 'ml-1m')

# step 2: read data
users = read_data(root / 'users.dat', sep='::', names=['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code'])
movies = read_data(root / 'movies.dat', sep='::', names=['MovieID', 'Title', 'Genres'])
ratings = read_data(root / 'ratings.dat', sep='::', names=['UserID', 'MovieID', 'Rating', 'Timestamp'])
