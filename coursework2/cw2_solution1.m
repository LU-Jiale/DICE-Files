% The weights just fail to converge if random maps are used. I have 
% discussed with some of my friends who taking RL as well, they have the same
% issues. Thus, I guess maybe the feature representation is not good enough 
% to let the weights converge. I was not able to solve this issue, so I have 
% to use the same map for training.

%% ACTION CONSTANTS:
UP_LEFT = 1 ;
UP = 2 ;
UP_RIGHT = 3 ;


%% PROBLEM SPECIFICATION:

blockSize = 5 ; % This will function as the dimension of the road basis 
% images (blockSize x blockSize), as well as the view range, in rows of
% your car (including the current row).

n_MiniMapBlocksPerMap = 5 ; % determines the size of the test instance. 

basisEpsisodeLength = blockSize - 1 ; % The agent moves forward at constant speed and
% the upper row of the map functions as a set of terminal states. So 5 rows
% -> 4 actions.

episodeLength = blockSize*n_MiniMapBlocksPerMap - 1 ;% Similarly for a complete
% scenario created from joining road basis grid maps in a line.

discountFactor_gamma = 0.7 ; % if needed

rewards = [ 1, -1, -20 ] ; % the rewards are state-based. 
% there is a significant reward penalty.

probabilityOfUniformlyRandomDirectionTaken = 0.15 ; % Noisy driver actions.

roadBasisGridMaps = generateMiniMaps ; % Generates the 8 road basis grid 
% maps, complete with an initial location for your agent. (Also see the 
% GridMap class).

noCarOnRowProbability = 0.8 ; % the probability that there is no car 
% spawned for each row

seed = 1234;
rng(seed); % setting the seed for the random nunber generator

% Call this whenever starting a new episode:
MDP = generateMap( roadBasisGridMaps, n_MiniMapBlocksPerMap, blockSize, ...
    noCarOnRowProbability, probabilityOfUniformlyRandomDirectionTaken, ...
    rewards );


%% Initialising the state observation (state features) and setting up the 
% exercise approximate Q-function:
stateFeatures = ones( 4, 5 );
action_values = zeros(1, 3);

Q_test1 = ones(4, 5, 3);
Q_test1(:,:,1) = 100;
Q_test1(:,:,3) = 100;% obviously this is not a correctly computed Q-function; it does imply a policy however: Always go Up! (though on a clear road it will default to the first indexed action: go left)

theta_test1 = zeros(4,5,3); % theta for state-action function

%% TEST ACTION TAKING, MOVING WINDOW AND TRAJECTORY PRINTING:
for episode = 1:5000000
    alpha = 1e-3 / sqrt(episode); % Learning step
    currentTimeStep = 0 ;
    rng(seed); % force to use same map
    MDP = generateMap( roadBasisGridMaps, n_MiniMapBlocksPerMap, ...
        blockSize, noCarOnRowProbability, ...
        probabilityOfUniformlyRandomDirectionTaken, rewards );
    currentMap = MDP ;
    agentLocation = currentMap.Start ;
    startingLocation = agentLocation ; % Keeping record of initial location.
        
    realAgentLocation = agentLocation ; % The location on the full test map.
    
    % list to store rewards, actions and states for the entire episode
    Return = [];
    actions = [];
    states = [];
    
    for i = 1:episodeLength
       
        stateFeatures = MDP.getStateFeatures(realAgentLocation);  % feature description of a state:
        states = [states; stateFeatures]; % append states to list
        for action = 1:3
            action_values(action) = ...
                sum ( sum( Q_test1(:,:,action) .* stateFeatures ) );
        end % for each possible action
        [~, actionTaken] = max(action_values);
        actions = [actions, actionTaken]; % append action to list
        [ agentRewardSignal, realAgentLocation, currentTimeStep, ...
            agentMovementHistory ] = ...
            actionMoveAgent( actionTaken, realAgentLocation, MDP, ...
            currentTimeStep, agentMovementHistory, ...
            probabilityOfUniformlyRandomDirectionTaken ) ;
        
        Return = [Return, agentRewardSignal]; % append reward to list
        
    end
    
    currentMap = MDP ;
    agentLocation = realAgentLocation ;
    
    % lists to store target rewards and estimated rewards
    Vts = [];
    Vps = [];
    
    for i = 1:episodeLength
        s = states(i*4-3:i*4,:);
        r = 0;
        a = actions(i);
        for j = i:episodeLength
            r = r + power(discountFactor_gamma, j-1)*Return(j);
        end % compute target reward
        Vps = [Vps, r]; % append target reward to list
        
        Q = theta_test1(:,:,a);
        Vt = sum(sum(Q.* s)); % compute estimated reward
        Vts = [Vts, Vt]; % append estimated reward to list
        Q = Q + alpha*(r - Vt)*s; 
        theta_test1(:,:,a) = Q; % update weights(theta_test1)
    end
    
    % Set throshold for training, moniter the training process every 1000 
    % epoches, stop training if MSE value is smaller and MSE differece is 
    % smaller than thresholds, or comes to the end of training epoches
    if mod(episode,1000)==0
        diff = Vps - Vts;
        diff = diff(:);
        MSE_pre = MSE;
        MSE = mean(power(diff,2))
        pause(1)
        if abs(MSE_pre - MSE)<0.0002 && MSE<0.1
            break;
        end
    end
    
end % for each episode
