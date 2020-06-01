from pathos.multiprocessing import ProcessingPool as pool
from tqdm import tqdm


def F(X,lamda=10,weight=0.05):
    print(X,lamda,weight)
 
 
    return res

zip_lamda = [ i for i in range(10) ]
x = [i  + 10 for i in range(10)]
 
with tqdm(total=len(x)) as t:
    for i, x in enumerate(pool.imap(F,x,zip_lamda)):
        t.update()

    pool.close()
    pool.join()