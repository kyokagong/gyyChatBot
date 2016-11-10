
import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_integer("num_gpus", 0,
                     "Total number of gpus for each machine."
                     "If you don't use GPU, please set it to '0'")
flags.DEFINE_string("device", "/gpu:0", "cpu mode or gpu mode")
flags.DEFINE_float("INIT_VAL", 0.1, "initial value for random initialization")
FLAGS = flags.FLAGS

class Singleton(object):
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, '_instance'):
            cls._instance = super(Singleton, cls).__new__(cls, *args, **kwargs)
        return cls._instance


"""a singleton for tensorflow session, so that i can use session anywhere by just one initialization"""
class SessionHandler(Singleton):
    def set_default(self):
        self.is_session_init = 1
        self._session = tf.InteractiveSession()


    def get_session(self):
        # if self.session is not None:
        #     self.session = tf.InteractiveSession()
        return self._session

    # initialize the variables have not been initialized
    def initialize_variables(self):
        v_list = tf.all_variables()
        for v in v_list:
            if not tf.is_variable_initialized(v).eval():
                self._session.run(v.initialized_value())

    def re_set(self):
        self._session.close()
        del self._session
        self.set_default()