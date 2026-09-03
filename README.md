# About Meridian GeoX

Geo experimentation (or incrementality testing) is a modern measurement
methodology that isolates the true causal impact of marketing campaigns by
running controlled tests across distinct geographic regions. Geo experiments
compare regions where a marketing variable is changed against regions where it
remains the same, anchoring your marketing strategy in causal reality and
measuring incremental lift independent of publisher attribution.

[Meridian GeoX](https://developers.google.com/meridian/geox) is an open-source
solution for measuring cross-publisher incrementality. By providing cookieless
and publisher-agnostic measurement, it helps close data gaps and generates
ground-truth experiment data used to calibrate Marketing Mix Models (MMMs). The
library offers cost-effective, multi-cell testing with flexible methodologies.
It integrates seamlessly with
[Meridian (MMM)](https://developers.google.com/meridian) to provide
the causal measurement needed to inform
[MMM priors](https://developers.google.com/meridian/docs/advanced-modeling/intro-priors).

Meridian GeoX helps you answer key questions such as:

*   What is the true incremental impact of my media spend across a specific
    platform or channel globally?
*   Which experiment design (such as holdback, go dark, or heavy up) is best
    suited to test my specific business objectives?
*   How can I use proven causal results to calibrate my Marketing Mix Models
    (MMM) and improve budget allocation?

## Core benefits

The revamped methodology in Meridian GeoX offers reliable incrementality testing
that is both cost-effective and time-saving. Benefits of using Meridian GeoX
include:

*   **Streamlined all-in-one solution**: Functions as a single library for all
    current and future methodologies, covering both study design and
    incrementality analysis. It also allows for comparing study designs across
    different methodologies.

*   **Tailored experiments**: Provides support for many types of experiment
    designs to support your needs, including holdback, go-dark, and heavy-up.

*   **Multi-cell capability**: Enables multi-cell execution, which saves both
    cost and time by comparing multiple treatment arms with a common control
    group.

*   **Flexible design**: Offers a flexible API to accommodate operational
    constraints or geo and statistical constraints in study design, such as
    forcing certain geos to be excluded from testing to avoid large media
    disruptions.

*   **Meridian MMM integration**: Seamlessly integrates with Meridian Marketing
    Mix Model (MMM), by suggesting new GeoX experiments for your Meridian model
    based on MMM results. These experiments help calibrate your model by using
    GeoX's open-source library to test and compare real-world data across
    different methodologies to provide incrementality-driven priors.

## Installation

### Prerequisites and system recommendations

**Python requirements**

*   Meridian GeoX only: Python >= 3.10 is required for Meridian GeoX
*   Meridian GeoX and Meridian MMM: Python >= 3.11 is required to use alongside
    Meridian MMM.

**GPU recommendations**

Meridian GeoX relies on **JAX** for high-performance vectorized computations.
Installing `meridian-geox` will automatically install CPU-based JAX. If you plan
to run heavy simulations or use JAX acceleration for both GeoX and Meridian, we
recommend setting up JAX with GPU support. Refer to the [JAX installation
guide](https://github.com/google/jax#installation) and the [Meridian
installation
guide](https://github.com/google/meridian/blob/main/README.md#install-meridian)
for GPU configurations.

### Install the Meridian GeoX library

To install `meridian-geox`, run the following command to automatically install
the most recent published version from PyPI:

```sh
$ pip install --upgrade meridian-geox
```

To install or upgrade both `google-meridian` and `meridian-geox` from PyPI, run:

```sh
$ pip install --upgrade 'google-meridian[meridian-geox]'
```

Alternatively, to install the development version directly from GitHub:

```sh
$ pip install --upgrade git+https://github.com/google/meridian-geox.git
```

## How to Use the Meridian GeoX Library

*   **Demo Colab**: Check the [Getting Started Colabs][10] for a complete
    interactive walkthrough of single-cell and multi-cell designs and analysis.
*   **Methodology**: Read the [research paper][11] for a deep dive into the
    library's core components, statistical methodology (Time-based Regression,
    etc).

## Meridian GeoX Documentation & Tutorials

For full technical documentation, mathematical formulations, and step-by-step
guides, visit the official [Meridian GeoX Documentation][1]:

| Resource                                   | Description                                                                    |
| ------------------------------------------ | ------------------------------------------------------------------------------ |
| [Meridian GeoX documentation][1]           | Main landing page for Meridian GeoX documentation.                             |
| [Meridian GeoX basics][2]                  | Learn about Meridian GeoX features and methodologies.                          |
| [Getting started colab][3]                 | Install and quickly learn how to use Meridian GeoX with this colab tutorial using sample data. |
| [Design][4]                                | A detailed walk-through of how to use the library to generate GeoX designs.    |
| [Implementation of geo testing][5]         | Step-by-step in-platform campaign configuration and best practices.            |
| [Analysis][6]                              | A detailed walk-through of how to use the library to generate GeoX analysis outputs. |
| [Data validation and quality checks][7]    | Learn about the built-in data validation and quality checks in Meridian GeoX.  |
| [MMM calibration][8]                       | Synthesizing GeoX results into ROI priors for Google Meridian MMM.             |
| [API reference][9]                         | Complete public API reference for all classes, methods, and parameters.        |

[1]: https://developers.google.com/meridian/geox
[2]: https://developers.google.com/meridian/geox/intro-to-geox
[3]: https://developers.google.com/meridian/geox/notebook
[4]: https://developers.google.com/meridian/geox/intro-to-design
[5]: https://developers.google.com/meridian/geox/implementation-of-geo-testing
[6]: https://developers.google.com/meridian/geox/intro-to-analysis
[7]: https://developers.google.com/meridian/geox/data-validation-and-quality-checks
[8]: https://developers.google.com/meridian/docs/advanced-modeling/set-custom-priors-past-experiments
[9]: https://developers.google.com/meridian/geox/api-reference
[10]: https://github.com/google/meridian-geox/tree/main/meridian_geox/colab
[11]: https://research.google/pubs/pub1090193/

## Support

**Questions about methodology**: See [Design methodology](https://developers.google.com/meridian/geox/design-methodology)
in the technical documentation.

**Issues installing or using Meridian GeoX**: Feel free to post questions in the
[Discussions](https://github.com/google/meridian-geox/discussions) or
[Issues](https://github.com/google/meridian-geox/issues) tabs of the GeoX
GitHub repository. The GeoX team responds to these questions weekly in batches,
so please be patient and don't reach out directly to your Google Account teams.

**Bug reports**: Please post bug reports to the
[Issues](https://github.com/google/meridian-geox/issues) tab of the GeoX GitHub
repository. We also encourage the community to share tips and advice with each
other on the [Issues](https://github.com/google/meridian-geox/issues) tab. When
our team addresses or resolves a new bug, we will notify you through the
comments on the issue.

**Feature requests**: Please post these to the
[Discussions](https://github.com/google/meridian-geox/discussions) tab of the
GeoX GitHub repository. We have an internal roadmap for GeoX development, but
welcome your inputs for new feature requests so that we can prioritize them
based on the roadmap.

**Pull requests**: These are appreciated but are very difficult for us to merge
because the code in this repository is linked to Google internal systems and has
to pass internal review. If you submit a pull request and we believe that we can
incorporate a change in the base code, we will reach out to you directly about
this.

## Citing Meridian GeoX

To cite this repository:

<!-- mdlint off(SNIPPET_INVALID_LANGUAGE) -->
```BibTeX
@software{meridian_geox_github,
  author = {Google Meridian GeoX Team},
  title = {Meridian GeoX: High-Performance Framework for Geographic Incrementality Experiments},
  url = {https://github.com/google/meridian-geox},
  version = {1.0.1},
  year = {2026},
}
```