# image_input_gym_env
For Gym environment, image processing as input

## Random action ###
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


## Result image ##


![Alt text](https://github.com/jkaewprateep/image_input_gym_env/blob/main/01.png?raw=true "Title")

![Alt text](https://github.com/jkaewprateep/image_input_gym_env/blob/main/78.png?raw=true "Title")

![Alt text](https://github.com/jkaewprateep/image_input_gym_env/blob/main/79.png?raw=true "Title")

![Alt text](https://github.com/jkaewprateep/image_input_gym_env/blob/main/80.png?raw=true "Title")
