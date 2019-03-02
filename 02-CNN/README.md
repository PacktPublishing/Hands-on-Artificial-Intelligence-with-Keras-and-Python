# Section 2: Convolutional Neural network, Autonomous drive in a driving simulator

In this project, you will use what you've learned about deep neural networks and convolutional neural networks to clone driving behavior. You will train, validate and test a model using Keras. The model will output a steering angle to an autonomous vehicle. Follow the steps below to run the project successfully. Please feel free to modify the pre-trained model and develop and train your own model.

* Download the Self-driving car simulator

  * [Linux](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5831f0f7_simulator-linux/simulator-linux.zip)

  * [Windows 32](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5831f4b6_simulator-windows-32/simulator-windows-32.zip) 

  * [Windows 64](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5831f3a4_simulator-windows-64/simulator-windows-64.zip)

I have downloaded the Windows version and port forwarded port 4567 from the Ubuntu VM to the WIndows VM for good graphical performance. However, feel free to download the Linux vesrion of the simulator.

* You can skip the next step if you want to use my pre-trained model. However, I didn't train for the whole track. I leave it as an exercise for you to navigate the car in the whole track. 

* Usage of `drive.py` requires you have saved the trained model as an h5 file, i.e. `model.h5`. See the [Keras documentation](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model) for how to create this file using the following command:
```sh
model.save(filepath)
```

* Once the model has been saved, it can be used with drive.py using this command:

```sh
python drive.py model.h5
```

* The above command will load the trained model and use the model to make predictions on individual images in real-time and send the predicted angle back to the server via a websocket connection.

* To train your model using the existing example dataset type the following command:

```sh
python model.py
```

This will create the model.h5 file containing all the weights of the learned model. And if you want to train with your own traing data put your downloaded data in the example folder or change the paths of the ```csvfile = './examples/driving_log.csv'``` and ```imgfolder = './examples/IMG/'```.

Note: There is known local system's setting issue with replacing "," with "." when using drive.py. When this happens it can make predicted steering values clipped to max/min values. If this occurs, a known fix for this is to add "export LANG=en_US.utf8" to the bashrc file.


