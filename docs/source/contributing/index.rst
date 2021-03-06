.. _contributing:

Contributing to Starfish
========================

If you're reading this section, you're probably interested in contributing to Starfish.  Welcome and thanks for your interest in contributing!

How can I contribute?
---------------------

Starfish is designed to specify pipeline recipes for image processing. To support this use, the library is composed as a series of `pipeline_component` modules.
The objects that sub-class `pipeline_component` spawn a command-line interface that should identically track the API of the python library.

A typical starfish run consists of running one or more image processing filter stages, and identifying features through either a spot- or pixel-based approach.
The identified features are then decoded into the genes that they correspond to by mapping the fluorescence channel (and optionally hybridization round) using a codebook.
Finally, the filtered data are segmented, identifying which cell each feature belongs to.

Creating a new algorithm for an existing `pipeline_component`
-------------------------------------------------------------

For example, to add a new image filter, one would:

1. Create a new python file `new_filter.py` in the `starfish/pipeline/filter/` directory.
2. Find the corresponding `AlgorithmBase` for your component.
   For filters, this is `FilterAlgorithmBase`, which is found in `starfish/pipeline/filter/_base.py`.
   Import that base into `new_filter.py`, and have your new algorithm subclass it,
   e.g. create `class NewFilter(FilterAlgorithmBase)`
3. Implement all required methods from the base class.
4. For the command line arguments to mimic the API, the arguments passed to the CLI must exactly
   match the names of the parameters for `NewFilter.__init__`, after dashes are automatically converted by argparse.
   For example, `--foo-bar` would convert to `foo_bar` and init must accept such an argument:
   `NewFilter.__init__(foo_bar, ..., **kwargs)`
5. `NewFilter.__init__()` must have a `**kwargs` parameter to accept arbitrary CLI args.

That's it! Your `NewFilter` algoritm will automatically register and be available under `starfish filter` in the CLI.
If at any point something gets confusing, it should be possible to look at existing pipeline components of the same
category for guidance on implementation.

Reporting bugs
--------------

- Bugs can be contributed as issues in the starfish repository.
  Please check to make sure there is no existing issue that describes the problem you
  have identified before adding your bug.
- When reporting issues please include as much detail as possible about your operating system,
  starfish version, slicedimage version, and python version. Whenever possible, please also include a brief,
  self-contained code example that demonstrates the problem, including a full traceback.

Code contributions
------------------

- Don't break the build: pull requests are expected to pass all automated CI checks.
  You can run those checks locally by running `make all` in starfish repository root.
- All Pull Request comments must be addressed, even after merge.
- All code must be reviewed by at least 1 other team member.
- All code must have typed parameters, and when possible, typed return calls (see
  `PEP484 <https://www.python.org/dev/peps/pep-0484>`_).
  We also encourage rational use of in-line variable annotation when the type of a newly defined object is not clear
  (see `PEP526 <https://www.python.org/dev/peps/pep-0526/>`_.
- All code must be documented according to `numpydoc <https://numpydoc.readthedocs.io/en/latest/>`_ style guidelines.
- Numpy provides an excellent `development workflow <https://docs.scipy.org/doc/numpy/dev/gitwash/development_workflow.html>`_
  that we encourage you to follow when developing features for starfish!
- All commits must have informative summaries; try to think about what will still make sense looking back on them next year.
- All pull requests should describe your testing plan.
- When merging a pull request, squash commits down to the smallest logical number of commits. In cases where a single commit
  suffices, use the "Squash and Merge" strategy, since it adds the PR number to the commit name. If multiple commits remain,
  use "Rebase and Merge".

Notebook contributions
----------------------

- All `.ipynb` files should have a corresponding `.py` file.
  Use `nbencdec <https://github.com/ttung/nbencdec>`_ to generate the corresponding `.py` file.
- The `.py` files allow refactor commands in the codebase to find code in the `.py` files,
  which is an important to keep the notebooks working as starfish evolves.

Debugging Errors
----------------

First, thank you for using Starfish and SpaceTx-Format! Feedback you provide on features and the
user experience is critical to making Starfish a successful tool. Because we iterate quickly on this
feedback to add new features, things change often, which can result in your code getting out of sync
with your data. When that happens, you may observe errors.

Most of the time, you can fix this problem by pulling the most recent version of the code,
reinstalling starfish, and restarting your environment. If you're using starfish with datasets from
spaceTx located on our cloudfront distribution, we're committed to keeping that data up to date.
Updated versions of the notebook will reference the correct data version, and copying over the
new link should fix any issues.

For example, if a notebook references in-situ sequencing data from August 23rd, and a breaking
change occurs on September 26th, it would be necessary to replace the experiment link to point at
data that was updated to work post-update:

.. code-block:: diff

    - http://spacetx.starfish.data.public.s3.amazonaws.com/browse/formatted/20180823/iss_breast/experiment.json
    + http://spacetx.starfish.data.public.s3.amazonaws.com/browse/formatted/20180926/iss_breast/experiment.json

If you're using your own data with starfish, you may need to re-run your data ingestion workflow
based on :py:class:`starfish.experiment.builder.providers.TileFetcher` and
:py:class:`starfish.experiment.builder.providers.FetchedTile` to generate up-to-date versions of spaceTx-format.

Upgrading to a new version
--------------------------

If you've installed from pypi, upgrading is as simple as reinstalling starfish.

.. code-block:: bash

    pip install --upgrade starfish

If you've installed our development version to take advantage of new features in real time, you'll
need to fetch changes and reinstall. Assuming you've cloned the respository into ``./starfish``,
you can install the newest version as follows:

.. code-block:: bash

    cd ./starfish
    git checkout master
    git pull
    pip3 install .

Reporting bugs
--------------

Bugs can be contributed as issues in the starfish repository. Please check to make sure there
is no existing issue that describes the problem you have identified before adding your bug.

When reporting issues please include as much detail as possible about your operating system,
starfish version, slicedimage version, and python version. Much of this can be accomplished by
sending us the output of ``pip freeze``:

.. code-block:: bash

    pip freeze > environment.txt

Whenever possible, please also include a brief, self-contained code example that demonstrates the
problem, including a full traceback.

We can also be contacted on the SpaceTx slack in the ``#starfish-users`` channel. If you'd like an
invitation, please feel free to email us. We can usually respond to bug reports same-day, and
are very appreciative of the time you take to submit them to us.
