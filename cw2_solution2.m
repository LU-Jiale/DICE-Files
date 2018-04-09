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
% Test instances are essentially road bases stacked one on top of the
% other.

basisEpsisodeLength = blockSize - 1 ; % The agent moves forward at constant speed and
% the upper row of the map functions as a set of terminal states. So 5 rows
% -> 4 actions.

episodeLength = blockSize*n_MiniMapBlocksPerMap - 1 ;% Similarly for a complete
% scenario created from joining road basis grid maps in a line.

rewards = [ 1, -1, -20 ] ; % the rewards are state-based. In order: paved 
% square, non-paved square, and car collision. Agents can occupy the same
% square as another car, and the collision does not end the instance, but
% there is a significant reward penalty.

probabilityOfUniformlyRandomDirectionTaken = 0.0 ; % Noisy driver actions.
% An action will not always have the desired effect. This is the
% probability that the selected action is ignored and the car uniformly 
% transitions into one of the above 3 states. If one of those states would 
% be outside the map, the next state will be the one above the current one.

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
% Q_test1(:,:,1) = 100;
% Q_test1(:,:,3) = 100;% obviously this is not a correctly computed Q-function; it does imply a policy however: Always go Up! (though on a clear road it will default to the first indexed action: go left)

theta = zeros(4,5,3); % theta for state-action function
eps = 0.2;

%% TEST ACTION TAKING, MOVING WINDOW AND TRAJECTORY PRINTING:
% Simulating agent behaviour when following the policy defined by 
% $pi_test1$.
%
% Commented lines also have examples of use for $GridMap$'s $getReward$ and
% $getTransitions$ functions, which act as our reward and transition
% functions respectively.
MSE = 0;
discountFactor_gamma = 0.9; % if needed
for episode = 1:5000
    alpha = 1e-5;
    currentTimeStep = 0 ;
    rng(seed);
    MDP = generateMap( roadBasisGridMaps, n_MiniMapBlocksPerMap, ...
        blockSize, noCarOnRowProbability, ...
        probabilityOfUniformlyRandomDirectionTaken, rewards );
    currentMap = MDP ;
    agentLocation = currentMap.Start ;
    startingLocation = agentLocation ; % Keeping record of initial location.
    
    % If you need to keep track of agent movement history:
    agentMovementHistory = zeros(episodeLength+1, 2) ;
    agentMovementHistory(currentTimeStep + 1, :) = agentLocation ;
        
    realAgentLocation = agentLocation ; % The location on the full test map.

    state_current = zeros(4,5);
    estimation_current = 0;
    lamda = 0.0; % for TD(0)
    theta_backup = theta;
    for i = 1:episodeLength-1
        e=ones(4,5); % for TD0
        % Use the $getStateFeatures$ function as below, in order to get the
        % feature description of a state:
        stateFeatures = MDP.getStateFeatures(realAgentLocation); % dimensions are 4rows x 5columns
        if randi(100) <= (1-100*eps)
            for action = 1:3
                action_values(action) = ...
                    sum ( sum( theta(:,:,action) .* stateFeatures ) );
            end % for each possible action
            [~, actionTaken] = max(action_values);
        else
            actionTaken = randi(3);
        end
        
        state_current = stateFeatures;
        action_current = actionTaken;
        
        Q = theta(:,:,action_current);
        estimation_current = Q(:)'*state_current(:);
        
        [ agentRewardSignal, realAgentLocation, currentTimeStep, ...
            agentMovementHistory ] = ...
            actionMoveAgent( actionTaken, realAgentLocation, MDP, ...
            currentTimeStep, agentMovementHistory, ...
            probabilityOfUniformlyRandomDirectionTaken ) ;
        
        reward_current = agentRewardSignal;
        
        % get the feature set of next step
        stateFeatures = MDP.getStateFeatures(realAgentLocation); % dimensions are 4rows x 5columns
        if randi(100) <= (1-100*eps)
            for action = 1:3
                action_values(action) = ...
                    sum ( sum( theta(:,:,action) .* stateFeatures ) );
            end % for each possible action
            [~, actionTaken] = max(action_values);
        else
            actionTaken = randi(3);
        end
        action_next = actionTaken;
        next_state = stateFeatures;
        
        % calculate Qt(s(t+1),a(t+1))
        Q = theta(:,:,action_next);
        estimation_next = Q(:)'*next_state(:);
        
        % Predict theta using TD0 
        delta = reward_current + discountFactor_gamma * ...
            estimation_next - estimation_current;
        Q = theta(:,:,action_current);
        Q = Q + alpha*delta*e;
        theta(:,:,action_current) = Q; % update theta

        e = discountFactor_gamma * lamda * e; % lambda =0, therefore e is reset to 0
              
    end
    
    currentMap = MDP ;
    agentLocation = realAgentLocation ;
    
    % Set throshold for training, moniter the training process every 1000 
    % epoches, stop training if  MSE differece betweeb new and old weights
    % is smaller than a threshold or comes to the end of training epoches
    if mod(episode,1000)==0
%         MSE_pre = MSE;
        SE = power(theta-theta_backup,2);
        MSE = mean(SE(:))
        if MSE <1e-13
            break;
        end
    end
end % for each episode


