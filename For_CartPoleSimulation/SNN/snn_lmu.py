import nengo
import numpy as np
import scipy.linalg
from scipy.special import legendre
import yaml
import os

#-----------------------------------------------------------------------------------------------------------------------

class NetInfo():
    def __init__(self):
        config = yaml.load(open(os.path.join('SI_Toolkit_ApplicationSpecificFiles', 'config.yml'), 'r'),
                       Loader=yaml.FullLoader)

        #self.ctrl_inputs = config['training_default']['control_inputs']         # For any reason, this is reading the full vector of [ctrl_in,state_in]
        self.ctrl_inputs = ['Q']                                                 # I force it, could not find a way to only read 'Q'
        self.state_inputs = config['training_default']['state_inputs']

        self.inputs = config['training_default']['control_inputs']
        self.inputs.extend(self.state_inputs)

        self.outputs = config['training_default']['outputs']

        self.net_type = 'SNN'

        # This part is forced to read from previous non-SNN network (should be changed when integrated properly to SI_Toolkit)
        self.path_to_normalization_info = './SI_Toolkit_ApplicationSpecificFiles/Experiments/Pretrained-RNN-1/Models/GRU-6IN-32H1-32H2-5OUT-0/NI_2021-06-29_12-02-03.csv'
        #self.parent_net_name = 'Network trained from scratch'
        self.parent_net_name = 'GRU-6IN-32H1-32H2-5OUT-0'
        #self.path_to_net = None
        self.path_to_net = './SI_Toolkit_ApplicationSpecificFiles/Experiments/Pretrained-RNN-1/Models/GRU-6IN-32H1-32H2-5OUT-0'

#-----------------------------------------------------------------------------------------------------------------------

def minmax(invalues, bound):
    '''Scale the input to [-1, 1]'''
    out =  2 * (invalues + bound) / (2*bound) - 1
    return out

#-----------------------------------------------------------------------------------------------------------------------

def scale_datasets(data, scales):
    '''Scale inputs in a list of datasets to -1, 1'''

    # Scale all datasets to [-1, 1] based on the maximum value found above
    bounds = []
    for var, bound in scales.items():
        bounds.append(bound)
    for ii in range(len(bounds)):
        data[:,ii] = minmax(data[:,ii], bounds[ii])

    return data

#-----------------------------------------------------------------------------------------------------------------------



class Delay:
    def __init__(self, dimensions, timesteps=50):
        self.history = np.zeros((timesteps, dimensions))

    def step(self, t, x):
        self.history = np.roll(self.history, -1)
        self.history[-1] = x
        return self.history[0]

# This code is completely taken from Terry Steward:
class DiscreteDelay(nengo.synapses.Synapse):
    def __init__(self, delay, size_in=1):
        self.delay = delay
        super().__init__(default_size_in=size_in, default_size_out=size_in)

    def make_state(self, shape_in, shape_out, dt, dtype=None, y0=None):
        return {}

    def make_step(self, shape_in, shape_out, dt, rng, state=None):
        steps = int(self.delay/dt)
        if steps == 0:
            def step_delay(t, x):
                return x
            return step_delay
        assert steps > 0

        state = np.zeros((steps, shape_in[0]))
        state_index = np.array([0])

        def step_delay(t, x, state=state, state_index=state_index):
            result = state[state_index]
            state[state_index] = x
            state_index[:] = (state_index + 1) % state.shape[0]
            return result

        return step_delay

# This code is completely taken from Terry Steward:
class LDN(nengo.Process):
    def __init__(self, theta, q, size_in=1):
        self.q = q  # number of internal state dimensions per input
        self.theta = theta  # size of time window (in seconds)
        self.size_in = size_in  # number of inputs

        # Do Aaron's math to generate the matrices A and B so that
        #  dx/dt = Ax + Bu will convert u into a legendre representation over a window theta
        #  https://github.com/arvoelke/nengolib/blob/master/nengolib/synapses/analog.py#L536
        A = np.zeros((q, q))
        B = np.zeros((q, 1))
        for i in range(q):
            B[i] = (-1.) ** i * (2 * i + 1)
            for j in range(q):
                A[i, j] = (2 * i + 1) * (-1 if i < j else (-1.) ** (i - j + 1))
        self.A = A / theta
        self.B = B / theta

        super().__init__(default_size_in=size_in, default_size_out=q * size_in)

    def make_step(self, shape_in, shape_out, dt, rng, state=None):
        state = np.zeros((self.q, self.size_in))

        # Handle the fact that we're discretizing the time step
        #  https://en.wikipedia.org/wiki/Discretization#Discretization_of_linear_state_space_models
        Ad = scipy.linalg.expm(self.A * dt)
        Bd = np.dot(np.dot(np.linalg.inv(self.A), (Ad - np.eye(self.q))), self.B)

        # this code will be called every timestep
        def step_legendre(t, x, state=state):
            state[:] = np.dot(Ad, state) + np.dot(Bd, x[None, :])
            return state.T.flatten()

        return step_legendre

    def get_weights_for_delays(self, r):
        # compute the weights needed to extract the value at time r
        # from the network (r=0 is right now, r=1 is theta seconds ago)
        r = np.asarray(r)
        m = np.asarray([legendre(i)(2 * r - 1) for i in range(self.q)])
        return m.reshape(self.q, -1).T


class DataParser:
    def __init__(self, data_df, sample_freq, vars=[]):
        self.data_df = data_df
        self.sample_freq = sample_freq
        self.vars = vars

    def parse_data(self, t):
        r = [self.data_df[x].iloc[int(t * self.sample_freq)] for x in self.vars]
        return r

    def update_data(self, data_df):
        self.data_df = data_df

class SwitchNode:
    def __init__(self, t_switch=4.0, stim_size=4):
        self.t_switch = t_switch
        self.stim_size = stim_size

    def step(self, t, x):
        if t <= self.t_switch:
            state = x[:self.stim_size]
        else:
            state = x[self.stim_size:]

        """
        cos = state[0]
        sin = state[1]
        norm = sin**2 + cos**2
        state[0] /= norm
        state[1] /= norm
        """
        state = np.clip(state, -1, 1)

        return state

class ErrorSwitchNode:
    def __init__(self, t_init=0.1, t_switch=4.0, stim_size=4, error="normal"):
        self.t_init = t_init
        self.t_switch = t_switch
        self.stim_size = stim_size
        self.error = error
        if error.lower() == "normal":
            self.error_func = self.normal
        elif error.lower() == "mse":
            self.error_func = self.mse
        else:
            raise ValueError('This error type is not supported. Please use "mse" or "normal".')

    def mse(self, a, b):
        return np.sign(a - b) * (a - b) ** 2

    def normal(self, a, b):
        return a - b

    def step(self, t, x):
        if self.t_init <= t <= self.t_switch:
            return self.error_func(x[:self.stim_size], x[self.stim_size:])
        return np.zeros(self.stim_size)

class PredictiveModelAutoregressiveLMU:
    def __init__(self, seed=42, neurons_per_dim=100, sample_freq=50,
                 lmu_theta=0.1, lmu_q=20, radius=1.5, dt=0.001,
                 t_delay=0.02, learning_rate=5e-5, action_vars=["Q"],
                 state_vars=["angle_sin", "angle_cos", "angleD", "position", "positionD"],
                 action_df=None, state_df=None, weights=None, scales={},
                 predict_delta=True, error="normal", t_switch=4.0, t_init=0.1,
                 extrapolation=False, *args, **kwargs):

        self.seed = seed
        self.neurons_per_dim = neurons_per_dim
        self.sample_freq = sample_freq
        self.t_delay = t_delay
        self.learning_rate = learning_rate
        self.action_vars = action_vars
        self.action_dim = len(action_vars)
        self.state_vars = state_vars
        self.state_dim = len(state_vars)
        self.radius = radius
        self.dt = dt
        self.lmu_q = lmu_q
        self.lmu_theta = lmu_theta
        self.weights = weights
        self.scales = scales
        self.predict_delta = predict_delta
        self.t_switch = t_switch
        self.t_init = t_init
        self.error = error
        self.extrapolation = extrapolation


        if action_df is None:
            self.action_df = pd.DataFrame(
                np.zeros((1, len(action_vars) + 1)),
                columns=["time"] + action_vars,
            )

        if state_df is None:
            self.state_df = pd.DataFrame(
                np.zeros((1, len(state_vars) + 1)),
                columns=["time"] + state_vars,
            )

        self.action_parser = DataParser(
            data_df=self.action_df,
            sample_freq=self.sample_freq,
            vars=self.action_vars
        )

        self.state_parser = DataParser(
            data_df=self.state_df,
            sample_freq=self.sample_freq,
            vars=self.state_vars
        )

        # this function streams the state signal from file to node
        def action_stim_func(t):
            return self.action_parser.parse_data(t)

        # this function streams the state signal from file to node
        def state_stim_func(t):
            return self.state_parser.parse_data(t)

        self.action_stim_func = action_stim_func
        self.state_stim_func = state_stim_func

        self.model, self.recordings = self.make_model()
        self.sim = nengo.Simulator(self.model, dt=self.dt, progress_bar=False)

    def set_inputs(self, action_df, state_df):

        for v in self.action_vars + ["time"]:
            assert v in action_df.columns
        for v in self.state_vars:
            assert v in state_df.columns

        self.action_df = action_df
        self.state_df = state_df

        self.action_parser.update_data(self.action_df)
        self.state_parser.update_data(self.state_df)

    def reset_sim(self):

        self.sim.reset(seed=self.seed)

    def set_weights(self, weights):

        weights = np.array(weights)
        assert weights.shape == (
            self.state_dim,
            self.neurons_per_dim * (self.state_dim + self.action_dim) * (1 + self.lmu_q)
        )

        self.weights = weights
        self.sim.signals[self.sim.model.sig[self.learned_connection]["weights"]] = self.weights

    def get_weights(self):

        weights = self.sim.signals[self.sim.model.sig[self.learned_connection]["weights"]]
        return np.array(weights)

    def process_files(self):

        t_max = self.action_df["time"].max()  # number of seconds to run
        self.sim.run(t_max)

        return self.recordings

    def get_state_dict(self):

        state_dict = {
            "seed": self.seed,
            "neurons_per_dim": self.neurons_per_dim,
            "sample_freq": self.sample_freq,
            "t_delay": self.t_delay,
            "learning_rate": self.learning_rate,
            "action_vars": self.action_vars,
            "state_vars": self.state_vars,
            "radius": self.radius,
            "dt": self.dt,
            "lmu_q": self.lmu_q,
            "lmu_theta": self.lmu_theta,
            "weights": self.get_weights(),
            "scales": self.scales,
            "predict_delta": self.predict_delta,
            "error": self.error,
            "t_switch": self.t_switch,
            "t_init": self.t_init,
            "extrapolation": self.extrapolation,
        }

        return state_dict

    def save_state_dict(self, path="model_state.pkl"):

        state_dict = self.get_state_dict()
        with open(path, "wb") as f:
            pickle.dump(state_dict, f)

    def make_model(self):

        if self.weights is None:
            self.weights = np.zeros((
                self.state_dim,
                self.neurons_per_dim * (self.state_dim + self.action_dim) * (1 + self.lmu_q))
            )

        model = nengo.Network()
        with model:
            # set the default synapse to None (normal default is 0.005s)
            model.config[nengo.Connection].synapse = None

            # initialize input nodes
            true_action = nengo.Node(self.action_stim_func)
            true_state = nengo.Node(self.state_stim_func)

            # create a node that first outputs the true state and then switches to predicted state
            switch = SwitchNode(t_switch=self.t_switch, stim_size=self.state_dim)
            believed_state = nengo.Node(switch.step, size_in=self.state_dim*2, size_out=self.state_dim)
            nengo.Connection(true_state, believed_state[:self.state_dim])

            # make a node for the predicted future state
            predicted_future_state = nengo.Node(None, size_in=self.state_dim)

            ##### EXPERIMENTAL AND DOES NOT WORK QUITE RIGHT...

            if self.extrapolation:
                # create a memory for the past state
                state_memory = nengo.Node(None, size_in=self.state_dim*2)
                nengo.Connection(believed_state, state_memory[:self.state_dim])
                nengo.Connection(
                    state_memory[:self.state_dim],
                    state_memory[self.state_dim:],
                    synapse=DiscreteDelay(self.t_delay)
                )

                # node to calculate the difference between the current and the last state
                difference = nengo.Node(None, size_in=self.state_dim)
                nengo.Connection(state_memory[:self.state_dim], difference)
                nengo.Connection(state_memory[self.state_dim:], difference, transform=-1)

                # feed the difference of the last two state into the prediction as an additional bias
                nengo.Connection(difference, predicted_future_state)

            # make LMU unit
            ldn = nengo.Node(LDN(theta=self.lmu_theta, q=self.lmu_q, size_in=self.state_dim + self.action_dim))
            nengo.Connection(true_action, ldn[:self.action_dim])
            nengo.Connection(believed_state, ldn[self.action_dim:])

            # make the hidden layer
            ens = nengo.Ensemble(
                n_neurons=self.neurons_per_dim * (self.state_dim + self.action_dim)*(1+self.lmu_q),
                dimensions=(self.state_dim + self.action_dim)*(1+self.lmu_q),
                neuron_type=nengo.LIFRate(),
                seed=self.seed
            )
            nengo.Connection(true_action, ens[:self.action_dim])
            nengo.Connection(believed_state, ens[self.action_dim:self.action_dim+self.state_dim])
            nengo.Connection(ldn, ens[self.action_dim+self.state_dim:])

            # if wanted, have the model predict the difference from the last state instead
            if self.predict_delta:
                print("predicting only the difference between the current and next state")
                nengo.Connection(believed_state, predicted_future_state)

            # make the output weights we can learn
            self.learned_connection = nengo.Connection(
                ens.neurons,
                predicted_future_state,
                transform=self.weights,  # change this if you have pre-recorded weights to use
                seed=self.seed,
                learning_rule_type=nengo.PES(
                    learning_rate=self.learning_rate,
                    pre_synapse=DiscreteDelay(self.t_delay)  # delay the activity value when updating weights
                )
            )

            # this is what the network predicted the current state to be in the past
            predicted_current_state = nengo.Node(None, size_in=self.state_dim)
            nengo.Connection(predicted_future_state, predicted_current_state, synapse=DiscreteDelay(self.t_delay))
            nengo.Connection(predicted_current_state, believed_state[self.state_dim:])

            # compute the error by subtracting the current measurement from a delayed version of the prediction
            error_node = ErrorSwitchNode(
                t_init=self.t_init,
                t_switch=self.t_switch,
                stim_size=self.state_dim,
                error=self.error
            )
            prediction_error = nengo.Node(error_node.step, size_in=self.state_dim*2, size_out=self.state_dim)
            nengo.Connection(predicted_current_state, prediction_error[:self.state_dim])
            nengo.Connection(true_state, prediction_error[self.state_dim:])

            # apply the error to the learning rule
            nengo.Connection(prediction_error, self.learned_connection.learning_rule)

            recordings = {
                "states" : nengo.Probe(true_state),
                "actions": nengo.Probe(true_action),
                "delay": self.t_delay,
                "predicted_current_states": nengo.Probe(predicted_current_state),
                "predicted_future_states": nengo.Probe(predicted_future_state),
                "prediction_errors": nengo.Probe(prediction_error),
            }

        return model, recordings
# -----------------------------------------------------------------------------------------------------------------------

class Predictor(object):
    def __init__(self, action_init, state_init, weights=None, seed=42, n=100, samp_freq=50,
               t_delay=0.02, learning_rate=0.0, dt=0.01):
        self.model = nengo.Network()
        self.model.config[nengo.Connection].synapse = None

        self.a = np.array(action_init)
        self.s = np.array(state_init)

        self.dt = dt
        self.in_sz = (len(self.s) + len(self.a))

        self.weights = weights
        if self.weights is None:
            self.weights = np.zeros((len(self.s), n * self.in_sz))

        self.z_pred = np.zeros(weights.shape[0])

        with self.model:
            a = nengo.Node(lambda t: self.a)
            s = nengo.Node(lambda t: self.s)

            #def set_z_pred(t, x):
                #self.z_pred[:] = x

            # the value to be predicted (which in this case is just the first dimension of the input)
            z = nengo.Node(None, size_in=4)
            nengo.Connection(s, z)

            z_pred = nengo.Node(None, size_in=4)

            # make the hidden layer
            ens = nengo.Ensemble(n_neurons=n * 5, dimensions=5,
                                     neuron_type=nengo.LIF(), seed=seed)
            nengo.Connection(a, ens[0])
            nengo.Connection(s, ens[1:])

            # make the output weights we can learn
            conn = nengo.Connection(ens.neurons, z_pred,
                                    transform=weights,  # change this if you have pre-recorded weights to use
                                    learning_rule_type=nengo.PES(learning_rate=learning_rate,
                                                                pre_synapse=DiscreteDelay(t_delay)
                                                                # delay the activity value when updating weights
                                                                ))

            self.sim = nengo.Simulator(self.model, dt=self.dt, progress_bar=False)
            self.ens = ens

    def step(self, a, s):
        self.a[:] = a
        self.s[:] = s
        self.sim.run(self.dt)
        return self.z_pred

    def return_internal_states(self, key):
        return np.asarray(self.sim.signals[self.sim.model.sig[self.ens.neurons][key]])

    def set_internal_states(self, internal_states, key):
        self.sim.signals[self.sim.model.sig[self.ens.neurons][key]] = internal_states

    def reset(self):
        self.sim.reset()

# -----------------------------------------------------------------------------------------------------------------------

class Predictor_LMU(object):
    def __init__(self, action_init, state_init, weights=None, seed=42, n=100, samp_freq=50, lmu_theta=0.1, lmu_q=20,
    t_delay=0.02, learning_rate=0, radius=1.5, dt = 0.001):

        self.model = nengo.Network()
        self.model.config[nengo.Connection].synapse = None

        self.a = np.array(action_init)
        self.s = np.array(state_init)

        self.dt = dt
        self.in_sz = (len(self.s) + len(self.a))

        self.weights = weights
        if self.weights is None:
            self.weights = np.zeros((len(self.s), n * self.in_sz))

        self.z_pred = np.zeros(weights.shape[0])

        with self.model:
            a = nengo.Node(lambda t: self.a)
            s = nengo.Node(lambda t: self.s)

            def set_z_pred(t, x):
                self.z_pred[:] = x

            # the value to be predicted (which in this case is just the first dimension of the input)
            z = nengo.Node(None, size_in=len(self.s))
            nengo.Connection(s, z)

            z_pred = nengo.Node(set_z_pred, size_in=len(self.s))

            ldn = nengo.Node(LDN(theta=lmu_theta, q=lmu_q, size_in=self.in_sz))

            nengo.Connection(a, ldn[0])
            nengo.Connection(s, ldn[1:])

            # make the hidden layer
            ens = nengo.Ensemble(n_neurons=n * self.in_sz, dimensions=self.in_sz * lmu_q,
                                     neuron_type=nengo.LIF(), seed=seed)

            # How do I connect each lmu to one dimension of ens?
            nengo.Connection(ldn, ens)

            # make the output weights we can learn
            conn = nengo.Connection(ens.neurons, z_pred,
                                    transform=weights,  # change this if you have pre-recorded weights to use
                                    learning_rule_type=nengo.PES(learning_rate=learning_rate,
                                                                 pre_synapse=DiscreteDelay(t_delay)
                                                                 # delay the activity value when updating weights
                                                                 ))

            self.sim = nengo.Simulator(self.model, dt=self.dt, progress_bar=False)
            self.ens = ens

    def step(self, a, s):
        self.a[:] = a
        self.s[:] = s
        self.sim.run(self.dt)
        return self.z_pred

    def return_internal_states(self, key):
        return np.asarray(self.sim.signals[self.sim.model.sig[self.ens.neurons][key]])

    def set_internal_states(self, internal_states, key):
        self.sim.signals[self.sim.model.sig[self.ens.neurons][key]] = internal_states


    def reset(self):
        self.sim.reset()

# -----------------------------------------------------------------------------------------------------------------------

class Predictor_LMU2(object):
    def __init__(self, action_init, state_init, scaling_factors, weights=None, seed=42, n=100, samp_freq=50, lmu_theta=0.1, lmu_q=20,
    t_delay=0.02, learning_rate=0, radius=1.5, dt = 0.001, t_switch=1000):

        self.model = nengo.Network()
        self.model.config[nengo.Connection].synapse = None

        self.a = np.array(action_init)
        self.s = np.array(state_init)

        self.dt = dt
        self.state_dim = len(self.s)
        self.action_dim = len(self.a)
        self.in_sz = (len(self.s)+len(self.a))
        self.t_switch = t_switch

        self.weights = weights
        if self.weights is None:
            self.weights = np.zeros((len(self.s), n * self.in_sz * (1 + lmu_q)))
        print(weights.shape)
        self.z_pred = np.zeros(weights.shape[0])

        # self.scales = scales
        self.scaling_factors = scaling_factors
        with self.model:
            # a = nengo.Node(lambda t: self.a)
            # s = nengo.Node(lambda t: self.s)
            true_action = nengo.Node(lambda t: self.a)
            true_state = nengo.Node(lambda t: self.s)

            def set_z_pred(t, x):
                # print(x.shape)
                self.z_pred[:] = x

            #####
            switch = SwitchNode(t_switch=self.t_switch, stim_size=self.state_dim)
            believed_state = nengo.Node(switch.step, size_in=self.state_dim * 2, size_out=self.state_dim)
            nengo.Connection(true_state, believed_state[:self.state_dim])

            predicted_future_state = nengo.Node(set_z_pred, size_in=self.state_dim)

            # z_pred = nengo.Node(set_z_pred, size_in=weights.shape[0])

            ldn = nengo.Node(LDN(theta=lmu_theta, q=lmu_q, size_in=self.in_sz))

            nengo.Connection(true_action, ldn[:self.action_dim])
            nengo.Connection(believed_state, ldn[self.action_dim:])

            # make the hidden layer
            ens = nengo.Ensemble(n_neurons=n * self.in_sz * (1 + lmu_q), dimensions=self.in_sz * (1 + lmu_q),
                                 neuron_type=nengo.LIFRate(), seed=seed, radius=radius)

            # How do I connect each lmu to one dimension of ens?
            nengo.Connection(true_action, ens[:1])
            nengo.Connection(true_state, ens[1:self.in_sz])
            nengo.Connection(ldn, ens[self.in_sz:])

            # make the output weights we can learn
            nengo.Connection(believed_state, predicted_future_state)

            nengo.Connection(ens.neurons, predicted_future_state,
                                    transform=self.weights)

        self.sim = nengo.Simulator(self.model, dt=self.dt, progress_bar=False)
        self.ens = ens

    def step(self, a, s):
        self.a[:] = a
        # self.s[:] = scale_datasets(s,self.scales)
        self.s[:] = s #/self.scaling_factors
        self.sim.run(self.dt)
        return self.z_pred # *self.scaling_factors

    # def return_internal_states(self, key):
    #     return np.asarray(self.sim.signals[self.sim.model.sig[self.ens.neurons][key]])
    #
    # def set_internal_states(self, internal_states, key):
    #     self.sim.signals[self.sim.model.sig[self.ens.neurons][key]] = internal_states


    def reset(self):
        self.sim.reset()

# -----------------------------------------------------------------------------------------------------------------------