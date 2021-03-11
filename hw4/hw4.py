# For database
from keras.datasets import mnist
# General packages
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

### Packages for classifiers
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm

#######################
######## Main script ##
#######################
### Load training and test data (x), labels (y)
(train_x, train_y), (test_x, test_y) = mnist.load_data()

### Reduction by random sampling
# Training: Reduce down training samples by random permutation
# shuffled_idx = np.random.permutation(np.arange(train_y.shape[0]))[:60000]
# train_x = train_x[shuffled_idx, :, :]
# train_y = train_y[shuffled_idx]

# # Test: Reduce down test samples by random perm.
# shuffled_idx = np.random.permutation(np.arange(test_y.shape[0]))[:10000]
# test_x = test_x[shuffled_idx, :, :]
# test_y = test_y[shuffled_idx]

### Reshape to 2D array "data matrix"
X = train_x.reshape(train_x.shape[0], 28*28)
# Subtract row-wise mean
Xm = X - X.mean(axis=1, keepdims=True)

# Do SVD
u, s, vh = np.linalg.svd(Xm, full_matrices=False)

print(u.shape)
print(s.shape)
print(vh.shape)

### Project
Yc = np.dot(Xm, vh.T) # PC projection / expansion coefficients

### Two and three digit helper functions
def two_dgt_prep(n, m):
    # Training data
    idx_n = np.where(train_y == n)
    idx_m = np.where(train_y == m)
    
    # Data for digits n,m
    Ynm = np.concatenate([Yc[idx_n,:][0,:,:], Yc[idx_m,:][0,:,:]])
    ynm = np.concatenate([train_y[idx_n], train_y[idx_m]])
    
    # Dimension reduction
    cutoff = 50#Yc.shape[1]#300 # modal cutoff (convergence condition)
    Ynm = Ynm[:,:cutoff]
    
    # Test data
    idx_nt = np.where(test_y == n)
    idx_mt = np.where(test_y == m)
    
    xt = test_x.reshape(test_x.shape[0], 28*28)
        
    xt_nm = np.concatenate([xt[idx_nt,:][0,:,:], xt[idx_mt,:][0,:,:]])
    yt_nm = np.concatenate([test_y[idx_nt], test_y[idx_mt]])
    
    Xt = xt_nm - xt_nm.mean(axis=1, keepdims=True)
    
    Yct = np.dot(Xt, vh.T) # expansion coefficients
    Yct = Yct[:,:cutoff] # dimension reduction
    
    return Ynm, ynm, Yct, yt_nm
### Make awesome partial sum movies
# plt.figure()
# plt.imshow(Xm[25,:].reshape(28,28))
# plt.show()

# ims = []
# fig, ax = plt.subplots()
# for j in range(200):
#     x = np.zeros((28*28))
#     for i in range(j):
#         x += Yc[25,i]*vh[i,:]#Y[25,i]*vh[i,:]/(np.dot(vh[i,:], vh[i,:].T))
    
#     # Plot eigendecomposition
#     im = ax.imshow(x.reshape(28,28), cmap='gray', animated=True)
#     #im.set_title('With ' + str(j) + ' modes')
#     title = ax.text(0.5, 1.05, "Terms in partial sum: {}".format(j), size=plt.rcParams["axes.titlesize"],
#                     ha='center', transform=ax.transAxes, )
    
#     ims.append([im, title])
#     #plt.pause(0.1)

# an = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
# an.save('partial_sums_test.mp4', fps=10)

# quit()

### Spectrum plot
# plt.figure()
# plt.semilogy(s, 'o')
# plt.grid(True)
# plt.xlabel('SVD mode number')
# plt.ylabel('Amplitudes (singular values)')
# plt.tight_layout()

### Do 3D plot
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# for j in [3, 8]:#range(9):
#     idx = np.where(train_y == j)
#     #ax.scatter(Yc[idx, 1], Yc[idx, 2], Yc[idx, 3], 'o', label='Digit ' + str(j))
#     ax.scatter(u[idx, 1], u[idx, 2], u[idx, 3], 'o', label='Digit ' + str(j))
# ax.set_xlabel('Mode 1')
# ax.set_ylabel('Mode 2')
# ax.set_zlabel('Mode 3')
# plt.legend()#loc='best')

# plt.show()

### Classifiers for digit identification
#############################################################
############ Part One: LDA ##################################
#############################################################
# model = LinearDiscriminantAnalysis()

# # Two digit classification
# score = np.ones((10, 10))
# for i in range(10):
#     for j in range(10):
#         if j != i:
#             n = i
#             m = j
            
#             Ynm, ynm, Yct, yt_nm = two_dgt_prep(n, m)
#             # Fit model
#             model.fit(Ynm, ynm) # expansion coefficients and labels
            
#             # On test data
#             #prediction = model.predict(Yct)
#             #right = np.where(prediction == yt_nm)
#             #score[i,j] = yt_nm[right].shape[0]/yt_nm.shape[0]
            
#             # On training data
#             prediction = model.predict(Ynm)
#             right = np.where(prediction == ynm)
#             score[i,j] = ynm[right].shape[0]/ynm.shape[0]
            
# #             # Check out
# #             # plt.figure()
# #             # plt.plot(prediction, 'o')
# #             # plt.plot(yt_nm, '--', label='true')
# #             # plt.legend(loc='best')
# #             # plt.xlabel('Test mode')
# #             # plt.ylabel('Classified digit')
# #             # plt.title('Classification of ' + str(n) + ' vs. ' + str(m) + ', score = %3.2f'%(score[i,j]*100.0) + '%')
# #             # plt.grid(True)
# #             # plt.tight_layout()
# #             # #plt.savefig('2digit_lda_' + str(n) + str(m) + '.png')
# #             # plt.show()

# plt.figure()
# plt.imshow(score, cmap='gray')
# plt.colorbar()
# plt.title('Two digit classification, LDA on training data')
# plt.show()

############################################# Three digit LDA classification
# # Obtain labels for digits n,m,l
# n = 5
# m = 6
# l = 7
# idx_n = np.where(train_y == n)
# idx_m = np.where(train_y == m)
# idx_l = np.where(train_y == l)
# # Data for digits n,m
# Ynml = np.concatenate([Yc[idx_n,:][0,:,:], Yc[idx_m,:][0,:,:], Yc[idx_l,:][0,:,:]])
# ynml = np.concatenate([train_y[idx_n], train_y[idx_m], train_y[idx_l]])
# # Fit model
# model.fit(Ynml, ynml) # PCs and labels

# # Try it out
# idx_nt = np.where(test_y == n)
# idx_mt = np.where(test_y == m)
# idx_lt = np.where(test_y == l)

# xt = test_x.reshape(test_x.shape[0], 28*28)

# xt_nml = np.concatenate([xt[idx_nt,:][0,:,:], xt[idx_mt,:][0,:,:], xt[idx_lt,:][0,:,:]])
# yt_nml = np.concatenate([test_y[idx_nt], test_y[idx_mt], test_y[idx_lt]])

# Xt = xt_nml - xt_nml.mean(axis=1, keepdims=True)

# Yct = np.dot(Xt, vh.T)

# prediction = model.predict(Yct)

# right = np.where(prediction == yt_nml)
# score = yt_nml[right].shape[0]/yt_nml.shape[0]

# plt.figure()
# plt.plot(prediction, 'o')
# plt.plot(yt_nml, '--', label='true')
# plt.legend(loc='best')
# plt.xlabel('Test mode')
# plt.ylabel('Classified digit')
# plt.title('Classification of ' + str(n) + ' vs. ' + str(m) + ' vs. ' + str(l) + ', score = %3.2f'%(score*100.0) + '%')
# plt.grid(True)
# plt.tight_layout()
# plt.savefig('3digit_lda_' + str(n) + str(m) + str(l) + '.png')
# plt.show()


##################################################################################
######################## Decision tree analysis ##################################
##################################################################################
tree = DecisionTreeClassifier(random_state=0)#, max_leaf_nodes = 4)#random_state=0)

score = np.ones((10,10))
for i in range(10):
    for j in range(10):
        if i != j:
            n = i
            m = j
            Ynm, ynm, Yct, yt_nm = two_dgt_prep(n, m)
            # fit tree
            tree.fit(Ynm, ynm)
            
            # Predict (test data)
            #prediction = tree.predict(Yct)
            #right = np.where(prediction == yt_nm)
            #score[i,j] = yt_nm[right].shape[0]/yt_nm.shape[0]
            
            # Predict (training data)
            prediction = tree.predict(Ynm)
            right = np.where(prediction == ynm)
            score[i,j] = ynm[right].shape[0]/ynm.shape[0]

plt.figure()
plt.imshow(score, cmap='gray')
plt.colorbar()
plt.title('Two digit classification with tree, on training data')
plt.show()


# plt.figure()
# plt.plot(prediction, 'o')
# plt.plot(yt_nm, '--', label='true')
# plt.legend(loc='best')
# plt.xlabel('Test mode')
# plt.ylabel('Classified digit')
# plt.title('Tree classification of ' + str(n) + ' vs. ' + str(m) + ', score = %3.2f'%(score*100.0) + '%')
# plt.grid(True)
# plt.tight_layout()
# plt.savefig('2digit_tree_' + str(n) + str(m) + '.png')
# plt.show()

##################################################################################
######################## Support vector classifier ###############################
##################################################################################
# vector_machine = svm.SVC()

# score = np.ones((10,10))

# for i in range(10):
#     for j in range(10):
#         if i != j:
#             n = i
#             m = j
#             Ynm, ynm, Yct, yt_nm = two_dgt_prep(n, m)
#             vector_machine.fit(Ynm, ynm)

#             # on test data
#             #prediction = vector_machine.predict(Yct)
#             #right = np.where(prediction == yt_nm)
#             #score[i,j] = yt_nm[right].shape[0]/yt_nm.shape[0]
            
#             # on training data
#             prediction = vector_machine.predict(Ynm)
#             right = np.where(prediction == ynm)
#             score[i,j] = ynm[right].shape[0]/ynm.shape[0]
            

# plt.figure()
# plt.imshow(score, cmap='gray')
# plt.colorbar()
# plt.title('Two digit classification with SVM on training data')
# plt.show()
