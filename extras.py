import matplotlib.pyplot as plt
import numpy as np
def draw_sample(image, label, num):
    #fig = plt.figure()
    for i in range(0, num):
        sub = plt.subplot(5, 5, i + 1)
        sub.imshow(image[i].reshape(28, 28), 
                    interpolation= 'nearest')
        sub.axis('off')
        sub.title.set_text(label[i])
    plt.show()

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def cost(Theta1, 
         Theta2,
         input_layer_size,
         hidden_layer_size, 
         feat_mat,
         target_vec, 
         learning_rate = 0.0001,
         L= 3,
         num_iter= 20):
    J = 0

    m = target_vec.shape[0] #num of examples

    #do the following num_iter times:
    for i in range(0, num_iter):
        
        #--------------------------------------------------------------#
        #Part1: for-prop

        a_1 = feat_mat
        a_1 = np.append(np.ones((a_1.shape[0], 1)), a_1, axis=1)

        z_2 = np.dot(a_1, np.transpose(Theta1))
        a_2 = sigmoid(z_2)
        
        #add the ones column:
        a_2 = np.append(np.ones((a_2.shape[0], 1)), a_2, axis= 1)

        z_3 = np.dot(a_2, np.transpose(Theta2))
        h_theta = sigmoid(z_3)

        #labels one-hot encoded:
        labels = np.zeros((m, 10))

        for j in range(0, m):
            labels[j][target_vec[j]] = 1

        J = - (1 / m) * np.sum(np.multiply(labels, np.log(h_theta)) + 
                               np.multiply(1 - labels, np.log(1 - h_theta)))
        #regularize J:
        J += (L / (2 * m)) * (np.sum(np.square(Theta1)) - 
                              np.sum(np.square(Theta1[:, 1])) +
                              np.sum(np.square(Theta2)) - 
                              np.sum(np.square(Theta2[:, 1]))
                             )

        if i % 10 == 0:
            print(f"iteration:{i}, J = %.2f"%J)
        #--------------------------------------------------------------#
        #Part2: back prop 
        
        del_3 = h_theta - labels

        g_dash_z2 = np.multiply(a_2, (1 - a_2))
        del_2 = np.multiply(np.dot(del_3, Theta2), g_dash_z2)

        #remove del_2_0 column vector from del_2 matrix
        del_2 = del_2[:, 1:]

        Delta_2 = np.dot(np.transpose(del_3), a_2)
        Delta_1 = np.dot(np.transpose(del_2), a_1)

        #print(f"Theta_1{Theta1.shape}")
        #print(f"Theta_2{Theta2.shape}")
        #print(f"Delta_1:{Delta_1.shape}")
        #print(f"Delta_2:{Delta_2.shape}")
        #finally write the gradient terms:
        grad1 = (1 / m) * Delta_1
        grad2 = (1 / m) * Delta_2

        #add regularization:
        reg_1 = Theta1
        reg_1[:, 0] = 0

        reg_2 = Theta2
        reg_2[:, 0] = 0

        #print(f"reg_1 :{reg_1.shape}")
        #print(f"reg_2 :{reg_2.shape}")
        #print(f"grad1 :{grad1.shape}")
        #print(f"grad2 :{grad2.shape}")
       
        grad1 += (L / m) * reg_1
        grad2 += (L / m) * reg_2

        #do gradient descent on Thetas:
        Theta1 -= learning_rate * grad1
        Theta2 -= learning_rate * grad2


    return J, Theta1, Theta2           

def predict(image, 
            Theta1,
            Theta2):
    image = np.append(np.ones((1, 1)), image, axis= 1)
    z_2 = np.dot(image, np.transpose(Theta1))
    a_2 = sigmoid(z_2)
    a_2 = np.append(np.ones((1,1)), a_2, axis= 1)

    z_3 = np.dot(a_2, np.transpose(Theta2))
    h_theta = sigmoid(z_3)
    print(h_theta)
    return (np.argmax(h_theta) + 1)




