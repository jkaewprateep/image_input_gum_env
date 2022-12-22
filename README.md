# image_input_gym_env #
For Gym environment, image processing as input is one way you working with AI Deep learning. We puspose a technical to work with refresh rates and we can including grids and image scalings and augmentation and significants the data input for costs efficients with tasks in application.

#### Problem: #### 
👧💬 For problem study they using Pygame because it is easy for beginner and ready input data as messages but in some problem we using camera or images.
1. Input image is large and we need to reduce data input to speed up at the efforadable costs of the networks learning.
2. Grayscales image or convolution image remove of our input detail such as the car colors, create the second path road in image reflecting from the main and information leave convolution layer with random image has detail. You need to apply Dense layer ```Dense(3)``` not only for colors but your AI not loss because conv need to hold some value for next time update. ``` Output = { information, gress background, gress background, gress background, information, gress background ... } ```
3. One problem of games AI or image camera AI is the input is in large variances, you need to do data-preparation input that is not too different.
4. How to add input calculating information without mixed with the input image and easy handling⁉️

#### Solutions: ####
🧸💬 You can create your own data-preparation process and method but from website such as Stack Overflow mostly they had problem with input data that make the AI cannot learn at the learning rates as it should.
1. Image input reduce sizes that make game play with lower costs computation when significants information is remains.
2. With a single input output layer or model, add its results with previous results or use the residual networks.
3. When working with image and we need to add information into the dataset without repeating or lossless method we add by the side channels.

## Gym game environment ##

The game environment, updated with info and prob in Gym 0.26.0 we are using and we observed the possible actions from ```env.action_space``` and create input to the game by ```env.step([ -1.,  0.,  0. ])```

```
env = gym.make("CarRacing-v2", render_mode='human')
outputs = env.action_space        					# Box([-1.  0.  0.], 1.0, (3,), float32)
obs = env.reset()
observation, reward, done, info, prob = env.step([ -1.,  0.,  0. ])	
```

## Random action ###

After we observed the actions space we create possible actions inputs that possible to play the game player before we test programming running and allowed our AI to learn to play the game.

```
def random_action ( ) :

    global DATA
	
    temp = tf.random.normal([1, 3], 1, 0.2, tf.float32)
    action = [ round( float(temp[0][0]), 2 ), round( float(temp[0][1]), 2 ), round( float(temp[0][2]), 2 ) ]
	
    # 1 wheel : [ 0.5, 0.0, 0.0 ]
    # 2 engine : [ 0.0, 1.0, 0.0 ]
    # 3 breaks : [ 0.0, 0.0, 1.0 ]
	
    return action
```

## Update dataset ##

When working with image and we need to add information into the dataset without repeating or lossless method we add by the side channels.

```
def update_DATA( image, action ):
    global reward
    global step
    global gamescores
    global DATA
    global LABEL
	
    step = step + 1
	
    contrl = step + ( 5 * reward )
    coff_0 = 1
    coff_1 = 1
    coff_2 = 1
    coff_3 = 1
    coff_4 = 1
    coff_5 = 1
    coff_6 = 1
    coff_7 = 1
    coff_8 = 1
	
    coeff_row = tf.constant( [ contrl, coff_0, coff_1, coff_2, coff_3, coff_4, coff_5, coff_6, coff_7, coff_8 ], shape=( 10, 1 ), dtype=tf.float32 )
    coeff_row = tf.concat([ coeff_row, tf.zeros([ 1014, 1 ], dtype=tf.float32) ], axis=0)
    coeff_row = tf.reshape( coeff_row, ( 32, 32, 1 ) )
	
    image = tf.reshape( image, (32, 32, 1) )
    coeff_row = tf.concat([ image, coeff_row ], axis=2 )
	
    DATA_row = tf.reshape( coeff_row, shape=(1, 1, 32, 32, 2) )
	
    DATA = tf.experimental.numpy.vstack([DATA, DATA_row])
    DATA = DATA[-30:,:,:,:]
	
    LABEL = tf.experimental.numpy.vstack([LABEL, tf.constant(action, shape=(1, 1, 1, 1, 3))])
    LABEL = LABEL[-30:,:,:,:]
	
    DATA = DATA[-30:,:,:,:]
    LABEL = LABEL[-30:,:,:,:]
	
    return DATA, LABEL
```

## Predict action ###

This function is to create action from trained model and input as image and information in the same space as the ```env.action_space``` , sample: ```[ -1.,  0.,  0. ]```

```
def predict_action ( ) :

    global DATA
	
    predictions = model.predict( tf.squeeze(DATA) )
    result = tf.abs( predictions[0] ).numpy()
	
    return result
```

## Generate image inpur as channel shifting information ##

Image flashes input information, we add previous image to remains the information otherwise AI difficult to learn because information less than gress backgrounds that is property of conv networks we are working on residual networks of image augmentation.

```
action = predict_action( )
observation, reward, done, info, prob = env.step(action)

if game_global_step >= 25 :
    image = tf.image.resize(observation, [32, 32])
    image = tf.image.rgb_to_grayscale( tf.cast( tf.keras.utils.img_to_array( image ), dtype=tf.float32 ) )
    image = tf.expand_dims( image, axis=0 )
		
    image = tf.keras.layers.Normalization(mean=3., variance=2.)(image)
    image = tf.keras.layers.Normalization(mean=4., variance=6.)(image)
    image = tf.squeeze( image )

    if game_global_step % 4 == 0 :
        temp_image = previous_image_1 + image
        previous_image_1 = temp_image
    elif game_global_step % 4 == 1 :
        temp_image = previous_image_2 + image
        previous_image_2 = temp_image
    elif game_global_step % 4 == 2 :
        temp_image = previous_image_3 + image
        previous_image_3 = temp_image
    elif game_global_step % 4 == 3 :
        temp_image = previous_image_4 + image
        previous_image_4 = temp_image
		
    DATA, LABEL = update_DATA( image, action )
```

## Model ###

We use the same model as we using in other games or image catagorize problem, that is working before modify and adjust is possible.

```
model = tf.keras.models.Sequential([
	tf.keras.layers.InputLayer(input_shape=INPUT_DIMS),

	tf.keras.layers.Normalization(mean=3., variance=2.),
	tf.keras.layers.Normalization(mean=4., variance=6.),
	tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
	tf.keras.layers.MaxPooling2D((2, 2)),
	tf.keras.layers.Dense(128, activation='relu'),
	tf.keras.layers.Reshape((128, 225)),
	tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(96, return_sequences=True, return_state=False)),
	tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(96)),
	tf.keras.layers.Flatten(),
	tf.keras.layers.Dense(192, activation='relu'),
	tf.keras.layers.Dense(3),
])

model.summary()
```

## Result image ##

Input image, from game input image ( 94 x 94 x 3 )

![Alt text](https://github.com/jkaewprateep/image_input_gym_env/blob/main/01.png?raw=true "Title")

Data preparation as image ( 32 x 32, 2 ) 

![Alt text](https://github.com/jkaewprateep/image_input_gym_env/blob/main/78.png?raw=true "Title")

Data preparation as image ( 32 x 32, 2 ) 

![Alt text](https://github.com/jkaewprateep/image_input_gym_env/blob/main/79.png?raw=true "Title")

Data preparation as image ( 32 x 32, 2 ) 

![Alt text](https://github.com/jkaewprateep/image_input_gym_env/blob/main/80.png?raw=true "Title")

Data preparation as image with position extraction.

![Alt text](https://github.com/jkaewprateep/image_input_gym_env/blob/main/Figure_1.png?raw=true "Title")
