CytonRL: an Efficient Reinforcement Learning Open-source Toolkit Implemented in C++

Xiaolin Wang (xiaolin.wang@nict.go.jp, arthur.xlw@gmail.com)

================================================

1) Prerequest
	cuda >= 9.0 
	
	cudnn >= 7.0
	
	opencv  https://sourceforge.net/projects/opencvlibrary/
	
	The Arcade Learning Environment (ALE)  https://github.com/mgbellemare/Arcade-Learning-Environment

2) Compile
	
	make

3) Train

bin/cytonRl --env roms/breakout.bin --mode train --saveModel model/model --showScreen 1

or faster training as,

bin/cytonRl --env roms/breakout.bin --mode train --saveModel model/model --showScreen 0  

The second command turn off the game window which makes the program run much faster. The game window can be brought back by creating a empty file "mode/model.screen", or turned off again by deleting that file.

4) Test

bin/cytonRl --mode test --env roms/breakout.bin  --loadModel model/model.100000000 --showScreen 1

or using our trained model as

bin/cytonRl --mode test --env roms/breakout.bin  --loadModel model-trained/model --showScreen 1

or faster test as

bin/cytonRl --mode test --env roms/breakout.bin  --loadModel model-trained/model --showScreen 0

The game window can be brought back by creating a empty file "mode/model.screen", or turned off again by deleting that file.

================================================

If you are using our toolkit, please kindly cite our paper (available at doc/cytonRl.pdf).

@article{wang,2018cytonmt,

  title={CytonRL: an Efficient Reinforcement Learning Open-source Toolkit Implemented in C++},

  author={Wang, Xiaolin},

  booktitle={To appear},

  year={2018}

}

================================================


5) Usage

bin/cytonRl --help

A.L.E: Arcade Learning Environment (version 0.5.1)
[Powered by Stella]
Use -help for help screen.
Warning: couldn't load settings file: ./ale.cfg
version 1.0
--help :	explanation, [valid values] (default)
--batchSize :	the size of batch (32)
--dueling :	using dueling DQN, 0|1 (1)
--eGreedy :	e-Greedy  start_value : end_value : num_steps (1.0:0.01:5000000)
--env :	the rom of an Atari game (roms/seaquest.bin)
--gamma :	parameter in the Bellman equation (0.99)
--inputFrames :	the number of concatnated frames as input (4)
--learningRate :	the learning rate (0.0000625)
--learnStart :	the step of starting learning (50000)
--loadModel :	the file path for loading a model ()
--maxEpisodeSteps :	the maximun number of steps per episode (18000)
--maxSteps :	the maximun number of training steps (100000000)
--mode :	working mode, train|test (train)
--networkSize :	the ouput dimension of each hidden layer (32:64:64:512)
--optimizer :	the optimzier, SGD|RMSprop (RMSprop)
--priorityAlpha :	the alpha parameter of prioritized experience replay (0.6)
--priorityBeta :	the beta parameter of prioritized experience replay (0.4)
--progPeriod :	the period of showing training progress (10000)
--replayMemory :	the capacity of replay memory (1000000)
--saveModel :	the file path for saving models (model/model)
--savePeriod :	the period of saving models (1000000)
--showScreen :	whether show screens of playing Atari games, 0|1. Creating or deleting the model/model.screen file can change this setting during the running of the program. Note that hidding screens makes program run faster. (0)
--targetQ :	the period of copy the current network to the target network  (30000)
--testEGreedy :	the e-Greedy threshold of test (0.001)
--testEpisodes :	the number of test epidoes (100)
--testMaxEpisodeSteps :	the maximun number of steps per test episode (18000)
--testPeriod :	the period of test (5000000)
--updatePeriod :	the period of learn a batch (4)
