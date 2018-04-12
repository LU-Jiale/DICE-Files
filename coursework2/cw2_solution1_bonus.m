% The weights just fail to converge if random maps are used. I have 
% discussed with some of my friends who taking RL as well, they have the same
% issues. Thus, I guess maybe the feature representation is not good enough 
% to let the weights converge. I was not able to solve this issue, so I have 
% to use the same map for training.

%% 0 for MC, 1 for TD 
ALGORITHM = 0;

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

probabilityOfUniformlyRandomDirectionTaken = 0.15 ; % Noisy driver actions.
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
Q_test1(:,:,1) = 100;
Q_test1(:,:,3) = 100;% obviously this is not a correctly computed Q-function; it does imply a policy however: Always go Up! (though on a clear road it will default to the first indexed action: go left)

theta_test1 = zeros(4,5,3); % theta for state-action function

%% TD0 algorithm
if ALGORITHM == 1
    discountFactor_gamma = 0.9; % if needed
    MSE = 0;
    for episode = 1:50000
        alpha = 1e-3;
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
        e=0;
        lamda = 0.0; % for TD(0)
        square_err=[];
        action_current = 0;
        for i = 1:episodeLength-1

            % Use the $getStateFeatures$ function as below, in order to get the
            % feature description of a state:
            stateFeatures = MDP.getStateFeatures(realAgentLocation); % dimensions are 4rows x 5columns
            for action = 1:3
                action_values(action) = ...
                    sum ( sum( Q_test1(:,:,action) .* stateFeatures ) );
            end % for each possible action
            [~, actionTaken] = max(action_values);
            state_current = stateFeatures;
            action_current = actionTaken;

            Q = theta_test1(:,:,action_current);
            estimation_current = Q(:)'*state_current(:);

            [ agentRewardSignal, realAgentLocation, currentTimeStep, ...
                agentMovementHistory ] = ...
                actionMoveAgent( actionTaken, realAgentLocation, MDP, ...
                currentTimeStep, agentMovementHistory, ...
                probabilityOfUniformlyRandomDirectionTaken ) ;

            reward_current = agentRewardSignal;

            % get the feature set of next step
            stateFeatures = MDP.getStateFeatures(realAgentLocation); % dimensions are 4rows x 5columns
            for action = 1:3
                action_values(action) = ...
                    sum ( sum( Q_test1(:,:,action) .* stateFeatures ) );
            end % for each possible action
            [~, actionTaken] = max(action_values);
            action_next = actionTaken;
            next_state = stateFeatures;
            % calculate Qt(s(t+1),a(t+1))
            Q = theta_test1(:,:,action_next);
            estimation_next = Q(:)'*next_state(:);

            delta = reward_current + discountFactor_gamma * ...
                estimation_next - estimation_current;
            square_err = [square_err, delta*delta];
            e = discountFactor_gamma * lamda * e + state_current;
            Q = theta_test1(:,:,action_current);
            Q = Q + alpha*delta*e;
            theta_test1(:,:,action_current) = Q;


            % save current value

        end

        currentMap = MDP ;
        agentLocation = realAgentLocation ;

        % Set throshold for training, moniter the training process every 1000 
        % epoches, stop training if  MSE differece is smaller than a threshold
        if mod(episode,1000)==0
            MSE_pre = MSE;
            MSE = mean(square_err)
            if MSE <0.1 && abs(MSE_pre - MSE)<0.002
                break;
            end
        end
    end % for each episode

%% MC algorithm
elseif ALGORITHM == 0
    discountFactor_gamma = 0.7
    for episode = 1:5000000
        alpha = 1e-3 / sqrt(episode);
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
        Return = [];
        actions = [];
        states = [];

        for i = 1:episodeLength
            % Use the $getStateFeatures$ function as below, in order to get the
            % feature description of a state:
            stateFeatures = MDP.getStateFeatures(realAgentLocation); % dimensions are 4rows x 5columns
            states = [states; stateFeatures];
            for action = 1:3
                action_values(action) = ...
                    sum ( sum( Q_test1(:,:,action) .* stateFeatures ) );
            end % for each possible action
            [~, actionTaken] = max(action_values);
            actions = [actions, actionTaken];
            [ agentRewardSignal, realAgentLocation, currentTimeStep, ...
                agentMovementHistory ] = ...
                actionMoveAgent( actionTaken, realAgentLocation, MDP, ...
                currentTimeStep, agentMovementHistory, ...
                probabilityOfUniformlyRandomDirectionTaken ) ;

            %     MDP.getReward( ...
            %             previousAgentLocation, realAgentLocation, actionTaken )
            Return = [Return, agentRewardSignal];

        end

        currentMap = MDP ;
        agentLocation = realAgentLocation ;

        Vts = [];
        Vps = [];
        for i = 1:episodeLength
            s = states(i*4-3:i*4,:);
            r = 0;
            a = actions(i);
            for j = i:episodeLength
                r = r + power(discountFactor_gamma, j-1)*Return(j);
            end
            Vps = [Vps, r];


            Q = theta_test1(:,:,a);
            Vt = sum(sum(Q.* s));
            Vts = [Vts, Vt];
            Q = Q + alpha*(r - Vt)*s;
            theta_test1(:,:,a) = Q;
        end

        % moniter the training process every 1000 epoches
        if mod(episode,1000)==0
            diff = Vps - Vts;
            diff = diff(:);
            MSE_pre = MSE;
            MSE = mean(power(diff,2))
            pause(1)
            if abs(MSE_pre - MSE)<0.002 && MSE<0.1
                break;
            end
        end

    end % for each episode
end

