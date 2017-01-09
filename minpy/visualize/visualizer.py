"""Common visualization methods for Minpy. This is based on
code from
https://github.com/dmlc/mxnet/blob/master/python/mxnet/optimizer.py."""

import minpy.visualize.summary_ops as summary_ops

try:
    from minpy.visualize.writer import SummaryWriter
except ImportError:
    pass

class Visualizer(object):
    """Base class of all visualizers."""
    vis_registry = {}

    @staticmethod
    def register(klass):
        """Register visualizers to the visualizer factory"""
        assert(isinstance(klass, type))
        name = klass.__name__.lower()

        if name in Visualizer.vis_registry:
            print('WARNING: New visualizer %s.%s is overriding '
				              'existing visualizer %s.%s' %
				              (klass.__module__, klass.__name__,
				  	            Visualizer.vis_registry[name].__module__,
				  	            Visualizer.vis_registry[name].__name__))

        Visualizer.vis_registry[name] = klass
        return klass

    @staticmethod
    def create_visualizer(name, **kwargs):
        """Create an visualizer with specified name.

		Parameters
		----------
		name: str
			Name of required visualizer. Should be the name
			of a subclass of Visualizer. Case insensitive.

		kwargs: dict
			Parameters for optimizer

		Returns
		-------
		vis: Visualizer
			The result visualizer.
		"""
        if name.lower() in Visualizer.vis_registry:
            return Visualizer.vis_registry[name.lower()](
				            **kwargs)
        else:
            raise ValueError('Cannot find visualizer %s' % name)

# convenience wrapper for Visualizer.register
register = Visualizer.register

@register
class NoVisualize(Visualizer):
    """By default, no visualization.
    """
    def __init__(self, **kwargs):
        """Initialization, just pass.
    	"""
        pass

    def single_scalar_summary(self, *args):
        """Initialization, just pass.
    	"""
        pass

    def norm_summary(self, *args):
        """Initialization, just pass.
    	"""
        pass

    def close(self):
        """Initialization, just pass.
    	"""
        pass

@register
class Tensorboard(Visualizer):
    """Visualization with tensorboard -- a suite of web applications
	for inspecting and understanding TensorFlow runs.

	Parameters
	----------
	summaries_dir: str
		Log files will be created in the directory declared by summaries_dir.
	writers: list
		A list that contains the names of the writers to be created.
	"""

    def __init__(self, **kwargs):
        self.summaries_dir = kwargs.pop('summaries_dir', '/private/tmp/model_log')
        writers = kwargs.pop('writers', ['train', 'test'])
        self.writers = {}
        for name in writers:
            self.writers[name] = SummaryWriter(self.summaries_dir + '/' + name)

    def single_scalar_summary(self, writer, tag, step, value):
        """Add a single scalar to a log file.

		Parameters
		----------
		writer: str
			The writer that will add the scalar summary. Its name should be
			a key for self.writers.
		tag: str
			The tag for the scalar summary.
		step: int
			The integer that indicates at which step/time the scalar is obtained.
		value: A 'float', 'int', 'long',
			'minpy.array.Array' with only one element,
			or 'numpy.ndarray' with only one element.
		"""
        scalar_summary = summary_ops.scalar_summary(tag, value)
        self.writers[writer].add_summary(scalar_summary, step)

    def norm_summary(self, writer, tag, step, values):
        """Apply the squared L2-norm, i.e., sum squared scalars and add it to
		a log file.

		Parameters
		----------
		writer: str
			The writer that will add the scalar summary. Its name should be
			a key for self.writers.
		tag: str
			The tag for the scalar summary.
		step: int
			The integer that indicates at which step/time the scalar is obtained.
		values: A dictionary where the values could be
			'float', 'int', 'long',
			'minpy.array.Array' with only one element,
			or 'numpy.ndarray' with only one element.
		"""
        norm_sum = 0
        for key, value in values.items():
            norm_sum += ((value**2).asnumpy()).sum()
        scalar_summary = summary_ops.scalar_summary(tag, norm_sum)
        self.writers[writer].add_summary(scalar_summary, step)

    def close(self):
        """Close the writers in the end.
    	"""
        for name, writer in self.writers.items():
            writer.close()

create = Visualizer.create_visualizer
