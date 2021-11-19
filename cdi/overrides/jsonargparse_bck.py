# import inspect
# import sys

# from argparse import Namespace, ArgumentError
# from jsonargparse import ArgumentParser
# from jsonargparse.util import _dict_to_flat_namespace, ParserError


# class ArgumentParser(ArgumentParser):
#     def parse_known_args(self, args=None, namespace=None):
#         """Raises NotImplementedError to dissuade its use, since typos in configs would go unnoticed."""
#         caller = inspect.getmodule(inspect.stack()[1][0]).__package__
#         # if caller not in {'jsonargparse', 'argcomplete'}:
#         #     raise NotImplementedError('parse_known_args not implemented to dissuade its use, since typos in configs would go unnoticed.')
#         if args is None:
#             args = sys.argv[1:]
#         else:
#             args = list(args)

#         if namespace is None:
#             namespace = Namespace()

#         if caller == 'argcomplete':
#             namespace = _dict_to_flat_namespace(self._merge_config(self.get_defaults(nested=False, skip_check=True), namespace))

#         try:
#             namespace, args = self._parse_known_args(args, namespace)
#         except (ArgumentError, ParserError):
#             err = sys.exc_info()[1]
#             self.error(str(err))

#         return namespace, args
