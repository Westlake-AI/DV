#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Equal Earth Projection
======================
This is a :mod:`matplotlib` add-on that adds the Equal Earth Projection
described by Bojan Šavrič (@BojanSavric), Tom Patterson and Bernhard Jenny:

Abstract:
    "The Equal Earth map projection is a new equal-area pseudocylindrical
    projection for world maps. It is inspired by the widely used Robinson
    projection, but unlike the Robinson projection, retains the relative size
    of areas. The projection equations are simple to implement and fast to
    evaluate. Continental outlines are shown in a visually pleasing and
    balanced way."

* https://doi.org/10.1080/13658816.2018.1504949
* https://www.researchgate.net/publication/326879978_The_Equal_Earth_map_projection

This projection is similar to the `Eckert IV equal area projection
<https://en.wikipedia.org/wiki/Eckert_IV_projection>`_, but is 2-5x
faster to calculate. It is based on code from:

* https://matplotlib.org/gallery/misc/custom_projection.html

as well as code from @mbostock:

* https://beta.observablehq.com/@mbostock/equal-earth-projection


Requirements
------------
shapefile (from pyshp) is required to read the map data. This is available
from Anaconda, but must be installed first, from the command line::

    >>>conda install shapefile

Installation
------------
Only the `EqualEarth.py <https://github.com/dneuman/EqualEarth/blob/master/EqualEarth.py>`_
file is required. You can download the entire repository using the green "Clone
or download" button, or by clicking on the file link, then right-clicking on
the "Raw" tab to download the actual script. The script must be located in a
directory in your `PYTHONPATH <https://scipher.wordpress.com/2010/05/10/setting-
your-pythonpath-environment-variable-linuxunixosx/>`_ list to use it in
another program.

.. note:: Using the :func:`GeoAxes.DrawCoastline` (new in 2.0) function will
          create a ``maps`` folder in the same directory and download some maps
          (500kb) for drawing, the first time it is called.

New in This Version (2.0)
-------------------------
:func:`GeoAxes.DrawCoastlines`:
    World map data from `Natural Earth <https://www.naturalearthdata.com>`_
    will download into the ``maps`` folder in the same directory as the
    Equal Earth module, the first time this function is called. This is 500kb
    on disk, but is downloaded in .zip format and unzipped automatically. Other
    maps can be used if you supply the shape files. Once the axes is set up,
    you can draw the continents::

        >>>ax.DrawCoastlines(facecolor='grey', edgecolor='k', lw=.5)

:func:`GeoAxes.plot_geodesic` Great Circle (geodesic) lines:
    Navigation lines can be plotted using the shortest path on the globe. These
    lines take plot keywords and wrap around if necessary::

        >>>pts = np.array([[-150, 45], [150, 45]])
        >>>ax.plot_geodesic(pts, 'b:', linewidth=1, alpha=.8)

:func:`GeoAxes.DrawTissot`:
    Draw the Tissot Indicatrix of Distortion on the projection. This is a set
    of circles of equal size drawn on the projection, showing how the
    projection distorts objects at various positions on the map::

        >>>ax.DrawTissot(width=10.)

    See `the Wikipedia article <https://en.m.wikipedia.org/wiki/Tissot%27s_indicatrix>`_
    for more information.

Usage
-----
Importing the module causes the Equal Earth projection to be registered with
Matplotlib so that it can be used when creating a subplot::

    import matplotlib.pyplot as plt
    import EqualEarth
    longs = [-200, 100, 100, -200]
    lats = [40, 40, -40, 40]
    fig = plt.figure('Equal Earth Projection')
    ax = fig.add_subplot(111, projection='equal_earth', facecolor='lightblue')
    ax.plot(longs, lats)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

.. figure:: _static/Equal_Earth_Projection.png
   :align:  center

.. note:: ax.plot():

          Lines drawn by `ax.plot()` method are clipped by the projection if
          any portions are outside it due to points being greater than +/- 180°
          in longitude. If you want to show lines wrapping around, they must be
          drawn twice. The second time will require the outside points put back
          into the correct range by adding or subtracting 360 as required.

Note that the default behaviour is to take all data in degrees. If radians
are preferred, use the ``rad=True`` optional keyword in ``fig.add_subplot()``,
ie::

    ax = fig.add_subplot(111, projection='equal_earth', rad=True)

All plots must be done in radians at this point.

This example creates a projection map with coastlines using the default
settings, and adds a few shortest-path lines that demonstrate the wrap-around
capabilities::

    import matplotlib.pyplot as plt
    import EqualEarth
    fig = plt.figure('Equal Earth', figsize=(10., 6.))
    fig.clear()
    ax = fig.add_subplot(111, projection='equal_earth',
                         facecolor='#CEEAFD')
    ax.tick_params(labelcolor=(0,0,0,.25))  # make alpha .25 to lighten
    pts = np.array([[-75, 45],
                    [-123, 49],
                    [-158, 21],
                    [116, -32],
                    [32.5, -26],
                    [105, 30.5],
                    [-75, 45]])
    ax.DrawCoastlines(zorder=0)  # put land under grid
    ax.plot(pts[:,0], pts[:,1], 'ro', markersize=4)
    ax.plot_geodesic(pts, 'b:', lw=2)
    ax.grid(color='grey', lw=.25)
    ax.set_title('Equal Earth Projection with Great Circle Lines',
                 size='x-large')
    plt.tight_layout()  # make most use of available space
    plt.show()

.. figure:: _static/Equal_Earth.png
   :align:  center

Future
------
Ultimately, the Equal Earth projection should be added to the :mod:`cartopy`
module, which provides a far greater range of features.


@Author: Dan Neuman (@dan613)

@Version: 2.0

@Date: 13 Sep 2018

EqualEarth API
==============
"""

from __future__ import unicode_literals

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import Circle
from matplotlib.path import Path
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator, Formatter, FixedLocator
from matplotlib.transforms import Affine2D, BboxTransformTo, Transform
from matplotlib.projections import register_projection
import matplotlib.spines as mspines
import matplotlib.axis as maxis
import numpy as np

# Mapping support
from zipfile import ZipFile
import pathlib
import io
from urllib.request import urlopen
import shapefile  # available via:  conda install shapefile

rcParams = matplotlib.rcParams

# This example projection class is rather long, but it is designed to
# illustrate many features, not all of which will be used every time.
# It is also common to factor out a lot of these methods into common
# code used by a number of projections with similar characteristics
# (see geo.py).


class GeoAxes(Axes):
    """
    An abstract base class for geographic projections. Most of these functions
    are used only by :mod:`matplotlib`, however :func:`DrawCoastlines` and
    :func:`plot_geodesic` are useful for drawing the continents and navigation
    lines, respectively.
    """
    class ThetaFormatter(Formatter):
        """
        Used to format the theta tick labels.  Converts the native
        unit of radians into degrees and adds a degree symbol.
        """
        def __init__(self, rad, round_to=1.0):
            self._round_to = round_to
            self._rad = rad

        def __call__(self, x, pos=None):
            if self._rad: x = np.rad2deg(x)
            degrees = np.round(x / self._round_to) * self._round_to
            if rcParams['text.usetex'] and not rcParams['text.latex.unicode']:
                return r"$%0.0f^\circ$" % degrees
            else:
                return "%0.0f\N{DEGREE SIGN}" % degrees

    RESOLUTION = 75

    def __init__(self, *args, rad=True, **kwargs):
        self._rad = rad
        if self._rad:
            self._limit = np.pi * 0.5
        else:
            self._limit = 90.
        super().__init__(*args, **kwargs)

    def _init_axis(self):
        self.xaxis = maxis.XAxis(self)
        self.yaxis = maxis.YAxis(self)
        # Do not register xaxis or yaxis with spines -- as done in
        # Axes._init_axis() -- until GeoAxes.xaxis.cla() works.
        # self.spines['geo'].register_axis(self.yaxis)
        self._update_transScale()

    def cla(self):
        Axes.cla(self)

        self.set_longitude_grid(30)
        self.set_latitude_grid(15)
        self.set_longitude_grid_ends(75)
        self.xaxis.set_minor_locator(NullLocator())
        self.yaxis.set_minor_locator(NullLocator())
        self.xaxis.set_ticks_position('none')
        self.yaxis.set_ticks_position('none')
        self.yaxis.set_tick_params(label1On=True)
        # Why do we need to turn on yaxis tick labels, but
        # xaxis tick labels are already on?

        self.grid(rcParams['axes.grid'])

        lim = self._limit
        Axes.set_xlim(self, -lim * 2., lim * 2.)
        Axes.set_ylim(self, -lim, lim)

    def _set_lim_and_transforms(self):
        # A (possibly non-linear) projection on the (already scaled) data

        # There are three important coordinate spaces going on here:
        #
        #    1. Data space: The space of the data itself
        #
        #    2. Axes space: The unit rectangle (0, 0) to (1, 1)
        #       covering the entire plot area.
        #
        #    3. Display space: The coordinates of the resulting image,
        #       often in pixels or dpi/inch.

        # This function makes heavy use of the Transform classes in
        # ``lib/matplotlib/transforms.py.`` For more information, see
        # the inline documentation there.

        # The goal of the first two transformations is to get from the
        # data space (in this case longitude and latitude) to axes
        # space.  It is separated into a non-affine and affine part so
        # that the non-affine part does not have to be recomputed when
        # a simple affine change to the figure has been made (such as
        # resizing the window or changing the dpi).

        # 1) The core transformation from data space into
        # rectilinear space defined in the EqualEarthTransform class.
        self.transProjection = self._get_core_transform(self.RESOLUTION)

        # 2) The above has an output range that is not in the unit
        # rectangle, so scale and translate it so it fits correctly
        # within the axes.  The peculiar calculations of xscale and
        # yscale are specific to an Equal Earth projection, so don't
        # worry about them too much.
        self.transAffine = self._get_affine_transform()

        # 3) This is the transformation from axes space to display
        # space.
        self.transAxes = BboxTransformTo(self.bbox)

        # Now put these 3 transforms together -- from data all the way
        # to display coordinates.  Using the '+' operator, these
        # transforms will be applied "in order".  The transforms are
        # automatically simplified, if possible, by the underlying
        # transformation framework.
        self.transData = \
            self.transProjection + \
            self.transAffine + \
            self.transAxes

        # The main data transformation is set up.  Now deal with
        # gridlines and tick labels.

        # Longitude gridlines and ticklabels.  The input to these
        # transforms are in display space in x and axes space in y.
        # Therefore, the input values will be in range (-xmin, 0),
        # (xmax, 1).  The goal of these transforms is to go from that
        # space to display space.  The tick labels will be offset 4
        # pixels from the equator.
        lim = self._limit # (pi/2 or 90°)
        self._xaxis_pretransform = \
            Affine2D() \
            .scale(1.0, lim * 2.0) \
            .translate(0.0, -lim)
        self._xaxis_transform = \
            self._xaxis_pretransform + \
            self.transData
        self._xaxis_text1_transform = \
            Affine2D().scale(1.0, 0.0) + \
            self.transData + \
            Affine2D().translate(0.0, 4.0)
        self._xaxis_text2_transform = \
            Affine2D().scale(1.0, 0.0) + \
            self.transData + \
            Affine2D().translate(0.0, -4.0)

        # Now set up the transforms for the latitude ticks.  The input to
        # these transforms are in axes space in x and display space in
        # y.  Therefore, the input values will be in range (0, -ymin),
        # (1, ymax).  The goal of these transforms is to go from that
        # space to display space.  The tick labels will be offset 4
        # pixels from the edge of the axes ellipse.
        yaxis_stretch = Affine2D().scale(lim * 4, 1).translate(-lim * 2, 0)
        yaxis_space = Affine2D().scale(1.0, 1.1)
        self._yaxis_transform = \
            yaxis_stretch + \
            self.transData
        yaxis_text_base = \
            yaxis_stretch + \
            self.transProjection + \
            (yaxis_space +
             self.transAffine +
             self.transAxes)
        self._yaxis_text1_transform = \
            yaxis_text_base + \
            Affine2D().translate(-8.0, 0.0)
        self._yaxis_text2_transform = \
            yaxis_text_base + \
            Affine2D().translate(8.0, 0.0)

    def _get_affine_transform(self):
        lim = self._limit
        transform = self._get_core_transform(1)
        xscale, _ = transform.transform_point((lim * 2, 0))
        _, yscale = transform.transform_point((0, lim))
        return Affine2D() \
            .scale(0.5 / xscale, 0.5 / yscale) \
            .translate(0.5, 0.5)

    def get_xaxis_transform(self, which='grid'):
        """
        Override this method to provide a transformation for the
        x-axis tick labels.

        Returns a tuple of the form (transform, valign, halign)
        """
        if which not in ['tick1', 'tick2', 'grid']:
            raise ValueError(
                "'which' must be one of 'tick1', 'tick2', or 'grid'")
        return self._xaxis_transform

    def get_xaxis_text1_transform(self, pad):
        return self._xaxis_text1_transform, 'bottom', 'center'

    def get_xaxis_text2_transform(self, pad):
        """
        Override this method to provide a transformation for the
        secondary x-axis tick labels.

        Returns a tuple of the form (transform, valign, halign)
        """
        return self._xaxis_text2_transform, 'top', 'center'

    def get_yaxis_transform(self, which='grid'):
        """
        Override this method to provide a transformation for the
        y-axis grid and ticks.
        """
        if which not in ['tick1', 'tick2', 'grid']:
            raise ValueError(
                "'which' must be one of 'tick1', 'tick2', or 'grid'")
        return self._yaxis_transform

    def get_yaxis_text1_transform(self, pad):
        """
        Override this method to provide a transformation for the
        y-axis tick labels.

        Returns a tuple of the form (transform, valign, halign)
        """
        return self._yaxis_text1_transform, 'center', 'right'

    def get_yaxis_text2_transform(self, pad):
        """
        Override this method to provide a transformation for the
        secondary y-axis tick labels.

        Returns a tuple of the form (transform, valign, halign)
        """
        return self._yaxis_text2_transform, 'center', 'left'

    def _gen_axes_patch(self):
        """
        Override this method to define the shape that is used for the
        background of the plot.  It should be a subclass of Patch.

        In this case, it is a Circle (that may be warped by the axes
        transform into an ellipse).  Any data and gridlines will be
        clipped to this shape.
        """
        return Circle((0.5, 0.5), 0.5)

    def _gen_axes_spines(self):
        return {'geo': mspines.Spine.circular_spine(self, (0.5, 0.5), 0.5)}

    def set_yscale(self, *args, **kwargs):
        if args[0] != 'linear':
            raise NotImplementedError

    # Prevent the user from applying scales to one or both of the
    # axes.  In this particular case, scaling the axes wouldn't make
    # sense, so we don't allow it.
    set_xscale = set_yscale

    # Prevent the user from changing the axes limits.  In our case, we
    # want to display the whole sphere all the time, so we override
    # set_xlim and set_ylim to ignore any input.  This also applies to
    # interactive panning and zooming in the GUI interfaces.
    def set_xlim(self, *args, **kwargs):
        raise TypeError("It is not possible to change axes limits "
                        "for geographic projections. Please consider "
                        "using Basemap or Cartopy.")

    set_ylim = set_xlim

    def format_coord(self, lon, lat):
        """
        Override this method to change how the values are displayed in
        the status bar.

        In this case, we want them to be displayed in degrees N/S/E/W.
        """
        if self._rad:
            lon, lat = np.rad2deg([lon, lat])
        if lat >= 0.0:
            ns = 'N'
        else:
            ns = 'S'
        if lon >= 0.0:
            ew = 'E'
        else:
            ew = 'W'
        return ('%f\N{DEGREE SIGN}%s, %f\N{DEGREE SIGN}%s'
                % (abs(lat), ns, abs(lon), ew))

    def set_longitude_grid(self, degrees):
        """
        Set the number of degrees between each longitude grid.

        This is an example method that is specific to this projection
        class -- it provides a more convenient interface to set the
        ticking than set_xticks would.
        """
        # Skip -180 and 180, which are the fixed limits.
        grid = np.arange(-180 + degrees, 180, degrees)
        if self._rad: grid = np.deg2rad(grid)
        self.xaxis.set_major_locator(FixedLocator(grid))
        self.xaxis.set_major_formatter(self.ThetaFormatter(self._rad, degrees))

    def set_latitude_grid(self, degrees):
        """
        Set the number of degrees between each longitude grid.

        This is an example method that is specific to this projection
        class -- it provides a more convenient interface than
        set_yticks would.
        """
        # Skip -90 and 90, which are the fixed limits.
        grid = np.arange(-90 + degrees, 90, degrees)
        if self._rad: grid = np.deg2rad(grid)
        self.yaxis.set_major_locator(FixedLocator(grid))
        self.yaxis.set_major_formatter(self.ThetaFormatter(self._rad, degrees))

    def set_longitude_grid_ends(self, degrees):
        """
        Set the latitude(s) at which to stop drawing the longitude grids.

        Often, in geographic projections, you wouldn't want to draw
        longitude gridlines near the poles.  This allows the user to
        specify the degree at which to stop drawing longitude grids.

        This is an example method that is specific to this projection
        class -- it provides an interface to something that has no
        analogy in the base Axes class.
        """
        if self._rad:
            self._longitude_cap = np.deg2rad(degrees)
        else:
            self._longitude_cap = degrees
        self._xaxis_pretransform \
            .clear() \
            .scale(1.0, self._longitude_cap * 2.0) \
            .translate(0.0, -self._longitude_cap)

    def get_data_ratio(self):
        """
        Return the aspect ratio of the data itself.

        This method should be overridden by any Axes that have a
        fixed data ratio.
        """
        return 1.0

    # Interactive panning and zooming is not supported with this projection,
    # so we override all of the following methods to disable it.
    def can_zoom(self):
        """
        Return *True* if this axes supports the zoom box button functionality.
        This axes object does not support interactive zoom box.
        """
        return False

    def can_pan(self):
        """
        Return *True* if this axes supports the pan/zoom button functionality.
        This axes object does not support interactive pan/zoom.
        """
        return False

    def start_pan(self, x, y, button):
        pass

    def end_pan(self):
        pass

    def drag_pan(self, button, key, x, y):
        pass

#=====================================================
#       Mapping Functions
#=====================================================
# iPython label
# %% Mapping

# These mapping functions will work with any projection based on GeoAxes

    _paths = ['maps/ne_110m_land/ne_110m_land',
          'maps/ne_110m_coastline/ne_110m_coastline',
          'maps/ne_110m_lakes/ne_110m_lakes']
    _names = ['land', 'coastline', 'lakes']


    def _CheckMaps(self, check_only=False):
        """
        Check to see if the maps already exist, otherwise download them from
        Natural Earth's content delivery network. It will be downloaded into the
        same directory as the EqualEarth module, in the 'maps' subdirectory.
        """
        url_template = ('http://naciscdn.org/naturalearth/110m'
                        '/physical/ne_110m_{name}.zip')
        path_template = 'ne_110m_{name}'
        p = pathlib.Path(__file__)
        pdir = p.parent
        print(pdir)
        mdir = pdir / 'maps'  # module maps directory
        if mdir.exists(): return True
        if check_only: return False

        # Now get the zip files
        mdir.mkdir()
        for name in self._names:
            url = url_template.format(name=name)
            mapdir = mdir / path_template.format(name=name)
            mapdir.mkdir()
            try:
                ne_file = urlopen(url)
                zfile = ZipFile(io.BytesIO(ne_file.read()), 'r')
                zfile.extractall(mapdir)
            finally:
                zfile.close()
        return True

    def _DrawEllipse(self, ll, width_deg, resolution=50):
        """
        Draw an ellipse. Technically, a circle is drawn (an
        ellipse with equal height and width), but this usually becomes
        an ellipse on the projection axes.

        Parameters
        ----------
        ll : tuple of floats
            longitude and latitude coordinates (in degrees) to draw the ellipse
        width_deg : float
            Width of ellipse in degrees
        resolution : int, optional, default: 50
            number of points to use in drawing the ellipse
        """
        # expect ll in degrees, so must
        # change ll to radians if that is the base unit
        if self._rad: ll = np.deg2rad(ll)
        long, lat = ll
        # Width as longitude range gets smaller as you go to the poles, so this
        # must be adjusted by the cosine of the latitude.
        if self._rad:
            height = np.deg2rad(width_deg)/2.  # use as radius, not diameter
            width = height/np.cos(lat)
        else:
            height = width_deg/2.
            width = height/np.cos(np.deg2rad(lat))
        # Use a path instead of the regular Ellipse patch to improve resolution
        t = np.linspace(0., 2. * np.pi, resolution)
        t = np.r_[t, [0]]  # append starting point to close path
        longs = width * np.cos(t) + long
        lats = height * np.sin(t) + lat
        verts = np.column_stack([longs, lats])
        patch = patches.Polygon(verts,
                                facecolor='r', alpha=.4,
                                edgecolor='none', zorder=5.)
        self.add_patch(patch)

    def DrawTissot(self, width=10., resolution=50):
        """
        Draw Tissot Indicatrices of Deformation over the map projection to show
        how the projection deforms equally-sized circles at various points
        on the map.

        Parameters
        ----------
        width : float, optional, default: 5.
            width of circles in degrees of latitude
        resolution : int, optional, default: 50
            Number of points in circle
        """
        degrees = 30
        for lat in range(-degrees, degrees+1, degrees):
            for long in range(-180, 181, degrees):
                self._DrawEllipse([long, lat], width, resolution)
        for lat in [-60, 60]:
            for long in range(-180, 181, 2*degrees):
                self._DrawEllipse([long, lat], width, resolution)
        for lat in [-90, 90]:
            self._DrawEllipse([0, lat], width, resolution)

    def DrawShapes(self, sf, **kwargs):
        """
        Draw shapes from the supplied shapefile. At the moment, only polygon
        and polyline shapefiles are supported, which are sufficient for
        drawing land-masses and coastlines. Coastlines are drawn separately
        from land-masses since the land-mass may have slices to allow internal
        bodies of water (e.g. Caspian Sea).

        Parameters
        ----------
        sf : shapefile.Reader object
            The shapefile containing the shapes to draw
        kwargs : optional
            Keyword arguments to send to the patch object. This will generally
            be edge and face colors, line widths, alpha, etc.
        """
        # Map points are in degrees, so must be converted if underlying
        # projection is in radians. Use a null function that does nothing
        # if the projection is in degrees.
        def null_convert(vals):
            return vals

        if self._rad:
            convert = np.deg2rad
        else:
            convert = null_convert

        if sf.shapeType == shapefile.POLYGON:
            for shape in sf.shapes():
                verts = convert(shape.points)
                patch = patches.Polygon(verts, **kwargs)
                self.add_patch(patch)
        elif sf.shapeType == shapefile.POLYLINE:
            for shape in sf.shapes():
                verts = convert(shape.points)
                path = patches.mlines.Path(verts)
                patch = patches.PathPatch(path, **kwargs)
                self.add_patch(patch)

    def DrawCoastlines(self, paths=None, edgecolor='k', facecolor='#FEFEE6',
                       linewidth=.25, **kwargs):
        """
        Draw land masses, coastlines, and major lakes. Colors and linewidth
        can be supplied. Coastlines are drawn separately from land-masses
        since the land-mass may have slices to allow internal bodies of water
        (e.g. Caspian Sea).

        Parameters
        ----------
        paths : list of str, optional, default: None
            List of paths to map data, if they aren't in the default location. The
            paths may be fully-specified or relative, and must be in order:
                ['land path', 'coastline path', 'lake path']
        edgecolor, ec : color, optional, default: black
            Color for coastlines and lake edges. ``ec`` can be used as a shortcut.
        facecolor, fc : color, optional, default: yellow
            Color for land. ``fc`` can be used as a shortcut.
        linewidth, lw : float, optional, default: .25
            Line width of coastlines and lake edges.
        """
        # Check that maps exist and download if necessary
        if not self._CheckMaps():
            print('maps not available')
            return

        # Set up colors, overriding defaults if shortcuts given
        bc = self.get_facecolor()         # background color
        ec = kwargs.pop('ec', edgecolor)  # edge color
        fc = kwargs.pop('fc', facecolor)  # face color
        lw = kwargs.pop('lw', linewidth)  # line width

        #        land   coast   lakes
        edges = ['none', ec,    ec]
        faces = [fc,    'none', bc]

        if not paths:
            paths = self._paths
        for path, f, e in zip(paths, faces, edges):
            sf = shapefile.Reader(path)
            self.DrawShapes(sf, linewidth=lw,
                            edgecolor=e, facecolor=f, **kwargs)

# %% Geodesic

    def Get_geodesic_heading_distance(self, ll1, ll2):
        """
        Return the heading and angular distance between two points. Angular
        distance is the angle between two points with Earth centre. To get actual
        distance, multiply the angle (in radians) by Earth radius. Heading is the
        angle between the path and true North.

        Math is found at http://en.wikipedia.org/wiki/Great-circle_navigation

        Parameters
        ----------
        ll1, ll2 : tuples of 2 floats
            start and end points as (longitude, latitude) tuples or lists
        """
        # Notation: *0 refers to node 0 where great circle intersects equator
        #           *1 refers to first point
        #           *01 refers to angle between node 0 and point 1

        # Heading is the angle between the path and true North.

        if not self._rad:
            ll1, ll2 = np.deg2rad((ll1, ll2))
        # simplify math notation
        cos = np.cos
        sin = np.sin
        atan = np.arctan2  # handles quadrants better than np.arctan
        # unpack longitudes and latitudes
        lon1, lat1 = ll1
        lon2, lat2 = ll2
        lon12 = lon2 - lon1  # longitudinal angle between the two points
        if lon12 > np.pi:
            lon12 -= np.pi * 2.
        elif lon12 < -np.pi:
            lon12 += np.pi * 2.

        y1 = cos(lat2) * sin(lon12)
        x1 = (cos(lat1) * sin(lat2)) - (sin(lat1) * cos(lat2) * cos(lon12))
        h1 = atan(y1, x1)  # heading of path

        y12 = np.sqrt((cos(lat1)*sin(lat2) - sin(lat1)*cos(lat2)*cos(lon12))**2 + \
                      (cos(lat2)*sin(lon12))**2)
        x12 = sin(lat1) * sin(lat2) + cos(lat1) * cos(lat2) * cos(lon12)
        d12 = atan(y12, x12)  # angular distance in radians
        if not self._rad:
            ll1 = np.rad2deg(ll1)
            h1, d12 = np.rad2deg((h1, d12))
        return ll1, h1, d12

    def Get_geodesic_waypoints(self, ll1, h1, d12):
        """
        Return an array of waypoints on the geodesic line given the start location,
        the heading, and the distance. The array will be in the native units
        (radians or degrees).

        Math is found at http://en.wikipedia.org/wiki/Great-circle_navigation

        Parameters
        ----------
        ll1 : tuple or list of floats
            The longitude and latitude of the start point
        h1 : float
            Heading (angle from North) from the start point
        d12 : float
            Angular distance to destination point
        """
        # Notation: *0 refers to node 0 where great circle intersects equator
        #           *1 refers to first point
        #           *01 refers to angle between node 0 and point 1

        # Angular distance is the angle between two points with Earth centre. To
        # get actual distance, multiply the angle (in radians) by Earth radius.

        # Heading is the angle between the path and true North.

        if not self._rad:
            ll1 = np.deg2rad(ll1)
            h1, d12 = np.deg2rad((h1, d12))
        lon1, lat1 = ll1
        # simplify math notation
        cos = np.cos
        sin = np.sin
        tan = np.tan
        atan = np.arctan2  # handles quadrants better than np.arctan
        # calculate where great circle crosses equator (node 0)
        y0 = sin(h1) * cos(lat1)
        x0 = np.sqrt(cos(h1)**2 + (sin(h1) * sin(lat1))**2)
        h0 = atan(y0, x0)  # heading at crossing point
        d01 = atan(tan(lat1), cos(h1))  # angular distance from node 0 to pt 1
        lon01 = atan(sin(h0) * sin(d01), cos(d01))
        lon0 = lon1 - lon01
        # create array of angular distances from node 0 to use
        ds = np.linspace(d01, d01+d12, self.RESOLUTION)
        # now calculate the latitudes and longitudes
        ys = cos(h0) * sin(ds)
        xs = np.sqrt(cos(ds)**2 + (sin(h0) * sin(ds))**2)
        lats = atan(ys, xs)
        lons = atan(sin(h0) * sin(ds), cos(ds)) + lon0
        if (np.abs(lons) > np.pi).any():  # check if any points outside map
            lons = (lons + np.pi) % (2. * np.pi) - np.pi
        result = np.column_stack([lons, lats])  # lons (x) go first
        if not self._rad: result = np.rad2deg(result)
        return result

    def Get_geodesic_points(self, ll1, ll2):
        """
        Return a list of arrays of points on the shortest path between
        two endpoints. Because the map wraps at +/- 180°, two arrays may be
        returned in the list.

        Parameters
        ----------
        ll1, ll2 : list-like
            (longitude, latitude) endpoints of the path
        """
        ll1, h1, d12 = self.Get_geodesic_heading_distance(ll1, ll2)
        verts = self.Get_geodesic_waypoints(ll1, h1, d12)

        # The map wraps around at +/- 180°, so the path must be broken up if path
        # wraps. Each part of the path must include one point outside the map
        # to make the path intersect with the border correctly.

        # return simple path if it doesn't wrap around
        # detect wrap by large change in longitude
        limit = 2. * self._limit
        diffs = verts[:-1, 0] - verts[1:, 0]  # change between each point and next
        i_chg, = (np.abs(diffs) > limit).nonzero()
        if len(i_chg) == 0:
            return [verts]

        # break into two parts, including a point for outside the map
        len1 = i_chg[0] + 1
        verts1 = verts[0:len1+1].copy()
        verts2 = verts[len1-1:].copy()

        # now fix the change points so they lie outside the map
        if verts1[0, 0] < 0:  # start is on the left (-ve)
            verts1[-1, 0] -= 2. * limit
            verts2[ 0, 0] += 2. * limit
        else:
            verts1[-1, 0] += 2. * limit
            verts2[ 0, 0] -= 2. * limit

        return [verts1, verts2]

    def plot_geodesic(self, *args, **kwargs):
        """
        Plot a geodesic path (shortest path on globe) between a series of points.
        The points must be given as (longitude, latitude) pairs, and there must
        be at least 2 pairs.

        Returns a list of lines.

        Parameters
        ----------
        data : array of floats
            The data may be an (n, 2) array of floats of longitudes and latitudes,
            or it may be as two separate arrays or lists of longitudes and
            latitudes, eg ``plot_geodesic(ax, lons, lats, **kwargs)``.
        *args : values to pass to the ax.plot() function
            These are positional, specifically the color/style string
        **kwargs : keyword arguments to pass to the ax.plot() function

        Examples
        --------
        Using two data styles::

            >>>longs = np.array([-70, 100, 100, -70])
            >>>lats = np.array([40, 40, -40, 40])
            >>>pts = np.column_stack([longs, lats])  # combine in (4,2) array
            >>>ax.plot_geodesic(longs, lats, 'b-', lw=1.)  # plot lines in blue
            >>>ax.plot_geodesic(pts, 'ro', markersize=4)   # plot points in red
        """
        results = []
        if (len(args) == 0):
            raise RuntimeError('No values were provided')

        a0 = np.array(args[0])
        args = args[1:]
        if len(a0.shape) == 1:  # have x values
            if len(args) == 0:  # but no y values
                raise RuntimeError('Need both x and y values')
            a1 = np.array(args.pop(0))
            args = args[1:]
            if len(a1.shape)==0:  # second arg not an array
                raise RuntimeError('Need both x and y values')
            points = np.column_stack([a0, a1])  # put x and y together
        elif len(a0.shape) == 2:  # have an (n, m) array
            if a0.shape[1] != 2:  # not an (n, 2) array
                raise ValueError('Must be an (n, 2) array or list')
            points = a0
        else:
            errmsg = ('Data points must be given as: '
                      'plot_geodesic(ax, lons, lats, *args, **kwargs) or ',
                      'plot_geodesic(ax, points, *args, **kwargs)')
            raise TypeError(errmsg)
        for i in range(len(points)-1):
            pts_list = self.Get_geodesic_points(points[i], points[i+1])
            for pts in pts_list:
                results.append(self.plot(pts[:,0], pts[:,1],
                                         *args, **kwargs))
        return results



class EqualEarthAxes(GeoAxes):
    """
    A custom class for the Equal Earth projection, an equal-area map
    projection, based on the GeoAxes base class.

    https://www.researchgate.net/publication/326879978_The_Equal_Earth_map_projection

    In general, you will not need to call any of these methods. Loading the
    module will register the projection with `matplotlib` so that it may be
    called using::

        >>>import matplotlib.pyplot as plt
        >>>import EqualEarth
        >>>fig = plt.figure('Equal Earth Projection')
        >>>ax = fig.add_subplot(111, projection='equal_earth')

    There are useful functions from the base :class:`GeoAxes` class,
    specifically:
        * :func:`GeoAxes.DrawCoastlines`
        * :func:`GeoAxes.plot_geodesic`, and
        * :func:`GeoAxes.DrawTissot`

    :func:`GeoAxes.DrawShapes` can also be useful to draw shapes if you
    provide a shapefile::

        >>>import shapefile
        >>>sf = shapefile.Reader(path)
        >>>ax.DrawShapes(sf, linewidth=.5, edgecolor='k', facecolor='g')

    At the moment :func:`GeoAxes.DrawShapes` only works with lines and
    polygon shapes.
    """

    # The projection must specify a name. This will be used by the
    # user to select the projection,
    # i.e. ``subplot(111, projection='equal_earth')``.
    name = 'equal_earth'

    def __init__(self, *args, rad=False, **kwargs):

        GeoAxes.__init__(self, *args, rad=rad, **kwargs)
        self._longitude_cap = self._limit
        self.set_aspect(0.5, adjustable='box', anchor='C')
        self.cla()

    def _get_core_transform(self, resolution):
        return self.EqualEarthTransform(resolution, self._rad)

    def _gen_axes_path(self):
        """
        Create the path that defines the outline of the projection
        """
        lim = self._limit
        verts = [(-lim * 2, -lim), # left, bottom
                 (-lim * 2,  lim), # left, top
                 ( lim * 2,  lim), # right, top
                 ( lim * 2, -lim), # right, bottom
                 (-lim * 2, -lim)] # close path

        return patches.Path(verts, closed=True)

    def _gen_axes_patch(self):
        """
        Override the parent method to define the shape that is used for the
        background of the plot.  It should be a subclass of Patch.

        In this case, it is a closed square path that is warped by the
        projection. Note that it must be in Axes space (0, 1).
        """
        path = self._gen_axes_path()  # Data space
        # convert to projection space with iterations on path
        ipath = self.transProjection.transform_path_non_affine(path)
        # convert to axes space
        apath = self.transAffine.transform_path(ipath)  # Axes space
        patch = patches.PathPatch(apath)
        return patch

    def _gen_axes_spines(self):
        """
        Generate the spine for the projection. This will be in data space.
        """
        spine_type = 'circle'
        path = self._gen_axes_path()
        spine = mspines.Spine(self, spine_type, path)
        return {'geo': spine}

    class EqualEarthTransform(Transform):
        """
        The base Equal Earth transform.
        """
        input_dims = 2
        output_dims = 2
        is_separable = False

        def __init__(self, resolution, rad):
            """
            Create a new Equal Earth transform.  Resolution is the number of
            steps to interpolate between each input line segment to approximate
            its path in curved Equal Earth space.
            """
            self._resolution = resolution
            self._rad = rad
            Transform.__init__(self)

        def transform_non_affine(self, ll):
            """
            Core transform, done in radians. Converts degree data to radians
            if self._rad is False.
            """
            if not self._rad: ll = np.deg2rad(ll)
            long, lat = ll.T

            # Constants
            A1 = 1.340264
            A2 = -0.081106
            A3 = 0.000893
            A4 = 0.003796
            A23 = A2 * 3.
            A37 = A3 * 7.
            A49 = A4 * 9.
            M = np.sqrt(3.)/2.
            p = np.arcsin(M * np.sin(lat))  # parametric latitude
            p2 = p**2
            p6 = p**6
            x = long * np.cos(p)/ \
                (M * (A1 + A23*p2 + p6*(A37 + A49*p2)))
            y = p*(A1 + A2*p2 + p6*(A3 + A4*p2))
            result = np.column_stack([x, y])
            if not self._rad: result = np.rad2deg(result)

            return result
        transform_non_affine.__doc__ = Transform.transform_non_affine.__doc__

        def transform_path_non_affine(self, path):
            # vertices = path.vertices
            ipath = path.interpolated(self._resolution)
            return Path(self.transform(ipath.vertices), ipath.codes)
        transform_path_non_affine.__doc__ = \
            Transform.transform_path_non_affine.__doc__

        def inverted(self):
            return EqualEarthAxes.InvertedEqualEarthTransform(self._resolution,
                                                              self._rad)
        inverted.__doc__ = Transform.inverted.__doc__

    class InvertedEqualEarthTransform(Transform):
        input_dims = 2
        output_dims = 2
        is_separable = False

        def __init__(self, resolution, rad):
            Transform.__init__(self)
            self._rad = rad
            self._resolution = resolution

        def transform_non_affine(self, xy):
            """
            Calculate the inverse transform using an iteration method, since
            the exact inverse is not solvable. Method based on
            https://beta.observablehq.com/@mbostock/equal-earth-projection
            """
            # if not using radians, convert from degrees first
            if not self._rad: xy = np.deg2rad(xy)
            x, y = xy.T
            # Constants
            iterations = 20
            limit = 1e-8
            A1 = 1.340264
            A2 = -0.081106
            A3 = 0.000893
            A4 = 0.003796
            A23 = A2 * 3.
            A37 = A3 * 7.
            A49 = A4 * 9.
            M = np.sqrt(3.)/2.
            # Use Newtons Method, where:
            #   fy is the function you need the root of
            #   dy is the derivative of the function
            #   dp is fy/dy or the change in estimate.
            p = y.copy()    # initial estimate for parametric latitude
            # Note y is a reference, so as p changes, so would y,
            # so make local copy, otherwise the changed y affects results
            dp = 0.  # no change at start
            for i in range(iterations):
                p -= dp
                p2 = p**2
                p6 = p**6
                fy = p*(A1 + A2*p2 + p6*(A3 + A4*p2)) - y
                dy = A1 + A23*p2 + p6*(A37 + A49*p2)
                dp = fy/dy
                if (np.abs(dp) < limit).all(): break
            long = M * x * dy/np.cos(p)
            lat = np.arcsin(np.sin(p)/M)
            result = np.column_stack([long, lat])
            if not self._rad: result = np.rad2deg(result)
            return result
        transform_non_affine.__doc__ = Transform.transform_non_affine.__doc__

        def inverted(self):
            return EqualEarthAxes.EqualEarthTransform(self._resolution,
                                                      self._rad)
        inverted.__doc__ = Transform.inverted.__doc__


# Now register the projection with matplotlib so the user can select it.
register_projection(EqualEarthAxes)

if __name__ == '__main__':
    fig = plt.figure('Equal Earth', figsize=(10., 6.))
    fig.clear()
    ax = fig.add_subplot(111, projection='equal_earth',
                         facecolor='#CEEAFD')
    # ax.tick_params(labelcolor=(0,0,0,.25))  # make alpha .25 to lighten
    pts = np.array([[-75, 45],
                    [-123, 49],
                    [-158, 21],
                    [116, -32],
                    [32.5, -26],
                    [105, 30.5],
                    [-75, 45]])
    # ax.DrawCoastlines(zorder=0)  # put land under grid
    ax.plot(pts[:,0], pts[:,1], 'ro', markersize=4)
    # ax.plot_geodesic(pts, 'b:', lw=2)
    # ax.grid(color='grey', lw=.25)
    # ax.set_title('Equal Earth Projection with Great Circle Lines',
    #              size='x-large')
    plt.tight_layout()  # make most use of available space
    plt.show()
    plt.savefig('equalearth.png', dpi=600)