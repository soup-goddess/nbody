#==========================================================
#ASTRONOMY COORDINATES AND UNITS
#==========================================================
import astropy.units as u
from astropy.coordinates import SkyCoord
import astropy.coordinates as coord

#==========================================================
#GALA - THE LIBRARY OF GALACTIC DYNAMICS
#==========================================================
import gala.potential as gal_pot        #Gravitational potentials
import gala.dynamics as gal_dyn         #Orbital integration and dynamics
from gala.dynamics import mockstream as ms  #Mock stream generator
from gala.units import galactic         #Galactic units system

#==========================================================
#MATH AND PLOTTING
#==========================================================
import pandas as pd

#standard plotting
import matplotlib.pyplot as plt

#interactive plotting
import plotly.express as px

#==========================================================
#DEFAULT PARAMETERS
#==========================================================
gala_modified = True

#==========================================================
#DATABASES USED
#==========================================================
'''
https://arxiv.org/pdf/2102.09568
https://people.smp.uq.edu.au/HolgerBaumgardt/globular/
'''
# ==========================================================
# INITIALIZATION OF CLUSTER DATA LIBRARY
# ==========================================================

class GlobularClusterLibrary:
    def __init__(self, data_file='globular_clusters.csv'): # Added data_file parameter
        self.data_file = data_file
        self.clusters_df = self._load_clusters_from_file() # Load data from file

    def _load_clusters_from_file(self):
        """Loads cluster data from the specified CSV file."""
        column_dtypes = {
            'name': str,
            'ra': float,
            'dec': float,
            'dis': float,
            'pm_ra_cosdec': float,
            'pm_dec': float,
            'rv': float,
            'mass': float
        }
        try:
            df = pd.read_csv(self.data_file)
            # Ensures column names match what Cluster_Data_initialization expects
            expected_cols = [
                'name', 'ra', 'dec', 'dis', 'pm_ra_cosdec', 'pm_dec', 'rv', 'mass'
            ]
            if not all(col in df.columns for col in expected_cols):
                raise ValueError(
                    f"CSV file must contain all expected columns: {expected_cols}"
                )
            return df
        # Returns statement telling user cluster is not within the library
        except FileNotFoundError:
            print(f"Warning: Cluster data file '{self.data_file}' not found. Starting with an empty library.")
            # If file doesn't exist, create an empty DataFrame with correct columns
            return pd.DataFrame(columns=[
                'name', 'ra', 'dec', 'dis', 'pm_ra_cosdec', 'pm_dec', 'rv', 'mass'
            ])
        # Other errors
        except Exception as e:
            print(f"Error loading clusters from '{self.data_file}': {e}")
            # Fallback to empty DataFrame on other errors
            return pd.DataFrame(columns=[
                'name', 'ra', 'dec', 'dis', 'pm_ra_cosdec', 'pm_dec', 'rv', 'mass'
            ])


    # Adds cluster to the current session of using the code, must call "save_clusters_to_file" in order to save permanently
    def add_cluster(self, name: str, ra: float, dec: float, dis: float,
                    pm_ra_cosdec: float, pm_dec: float, rv: float, mass: float, rrv: float, brv: float):
        # Check to see if cluster already exists in library
        if (self.clusters_df['name'].str.lower() == name.lower()).any():
            print(f"Warning: Cluster '{name}' already exists in the library. Not added again.")
            return

        # Add new cluster to the library
        new_cluster_data = pd.DataFrame([{
            'name': name, 'ra': ra, 'dec': dec, 'dis': dis,
            'pm_ra_cosdec': pm_ra_cosdec, 'pm_dec': pm_dec, 'rv': rv, 'mass': mass,
            'rrv': rrv, 'brv': brv
        }])
        self.clusters_df = pd.concat([self.clusters_df, new_cluster_data], ignore_index=True)
        print(f"Added '{name}' to the in-memory library.")

    # Save entered cluster to the library permanently
    def save_clusters_to_file(self):
        """Saves the current state of the in-memory cluster library back to the CSV file."""
        try:
            self.clusters_df.to_csv(self.data_file, index=False) # index=False prevents writing DataFrame index as a column
            print(f"Cluster library saved to '{self.data_file}'.")
        except Exception as e:
            print(f"Error saving clusters to '{self.data_file}': {e}")

    # Retrieving Globular Cluster data
    def get_cluster_data(self, cluster_name: str):
        cluster_info = self.clusters_df[self.clusters_df['name'].str.lower() == cluster_name.lower()]
        if cluster_info.empty:
            raise ValueError(f"Cluster '{cluster_name}' not found in the library. Available clusters: {self.list_clusters()}")
        info = cluster_info.iloc[0]
        return Cluster_Data_initialization(
            name=info['name'], ra=info['ra'], dec=info['dec'], dis=info['dis'],
            pm_ra_cosdec=info['pm_ra_cosdec'], pm_dec=info['pm_dec'],
            rv=info['rv'], mass=info['mass'], rrv=info['rrv'], brv=info['brv']
        )

    def list_clusters(self) -> list:
        return self.clusters_df['name'].tolist()

    # Deleting a cluster if needed
    def delete_cluster(self, cluster_name: str, save_to_file: bool = False):
        """
        Deletes all entries for a given cluster name from the in-memory library.
        If save_to_file is True, also saves the updated library to the CSV file.
        """
        initial_rows = len(self.clusters_df)
        # Use boolean indexing to keep only rows where the name does NOT match
        self.clusters_df = self.clusters_df[
            self.clusters_df['name'].str.lower() != cluster_name.lower()
            ].reset_index(drop=True)  # Reset index after dropping rows

        rows_deleted = initial_rows - len(self.clusters_df)
        if rows_deleted > 0:
            print(f"Deleted {rows_deleted} entries for '{cluster_name}' from the in-memory library.")
            if save_to_file:  # <-- This is the key to persistence
                self.save_clusters_to_file()
        else:
            print(f"No entries found for '{cluster_name}' to delete.")

# ==========================================================
# INITIALIZATION OF CLUSTER DATA
# ==========================================================


class Cluster_Data_initialization:

    def __init__(self, name, mass, ra, dec,
                 dis, pm_ra_cosdec, pm_dec, rv, brv, rrv):
        self.name = name  # Name of Globular Cluster
        self.mass = mass  # Mass
        self.ra = ra  # Right ascension
        self.dec = dec  # Declination
        self.dis = dis  # Distance from the sun in KiloParsecs
        self.pm_ra_cosdec = pm_ra_cosdec
        self.pm_dec = pm_dec
        self.rv = rv
        self.brv = brv
        self.rrv = rrv  # Radial Velocity
        self.astropy_units = (u.kpc, u.Myr, u.Msun, u.degree)

    # ==========================================================
    # IMPLEMENT OPTION FOR SHIFTED RADIAL VELOCITIES
    # ==========================================================
    def skycoord(self, shift_rv: float = None) -> SkyCoord:
        radial_velocity = self.rv
        if shift_rv is not None:
            radial_velocity = shift_rv
        # ==========================================================
        # SET ALL VARIABLES TO APPROPRIATE UNITS
        # ==========================================================
        return coord.SkyCoord(
            ra=self.ra * u.degree,
            dec=self.dec * u.degree,
            distance=self.dis * u.kpc,
            pm_ra_cosdec=self.pm_ra_cosdec * u.mas / u.yr,
            pm_dec=self.pm_dec * u.mas / u.yr,
            radial_velocity=radial_velocity * u.km / u.s,

        )


    # Determine the position of the given cluster
    def phase_space_position(self, skycoord_rv):
        sky_coord = self.skycoord(
            shift_rv=skycoord_rv,
        )

        self.galactocentric = sky_coord.transform_to(coord.Galactocentric)

        pos = [
            self.galactocentric.x.to(u.kpc).value,
            self.galactocentric.y.to(u.kpc).value,
            self.galactocentric.z.to(u.kpc).value
        ]

        vel = [
            self.galactocentric.v_x.to(u.km/u.s).value,
            self.galactocentric.v_y.to(u.km/u.s).value,
            self.galactocentric.v_z.to(u.km/u.s).value
        ]
        pos_units = pos * u.kpc
        vel_units = vel * u.km/u.s
        self.phase_space_pos = gal_dyn.PhaseSpacePosition(pos=pos_units, vel=vel_units)
        print(pos, vel)

        return self.phase_space_pos

    # Convert the given mass into correct units
    def get_progenitor_mass(self):
        """Returns the mass of the globular cluster with units."""

        return self.mass * u.Msun


# ==========================================================
# GALACTIC DATA AND STREAM SIMULATION
# ==========================================================

class Galactic_stream:

    # ==========================================================
    # SET INITIAL PARAMETERS
    # ==========================================================

    def __init__(self):
        self.potential = None
        self.Om_bar = None
        self.frame = None
        self.H = None
        self.df = None
        self.galactic_potentials()

    # ==========================================================
    # SET GALACTIC POTENTIALS
    # ==========================================================

    def galactic_potentials(self):

        # Galactic Bar uses Long-Murali Bar potential
        self.potential = gal_pot.CCompositePotential()
        self.potential['bar'] = gal_pot.LongMuraliBarPotential(
            m=2E10 * u.Msun,
            a=4 * u.kpc,
            b=0.5 * u.kpc,
            c=0.5 * u.kpc,
            alpha=25 * u.degree,
            units=galactic
        )
        # Galactic Disk uses Miyamoto-Nagai potential
        self.potential['disk'] = gal_pot.MiyamotoNagaiPotential(
            m=5E10 * u.Msun,
            a=3. * u.kpc,
            b=280. * u.pc,
            units=galactic
        )

        # Galactic Dark Matter Halo uses NFW potential
        self.potential['halo'] = gal_pot.NFWPotential(
            m=6E11 * u.Msun,
            r_s=20. * u.kpc,
            units=galactic
        )

        # Define rotating frame
        self.Om_bar = 42. * u.km / u.s / u.kpc
        self.frame = gal_pot.ConstantRotatingFrame(
            Omega=[0, 0, self.Om_bar.value] * self.Om_bar.unit,
            units=galactic
        )

        # Define the Hamiltonian
        self.H = gal_pot.Hamiltonian(
            potential=self.potential,
            frame=self.frame
        )

        # Using the Chen Stream
        self.df = ms.ChenStreamDF()


    # ==========================================================
    # STREAM PROJECTION DATA INITIALIZATION
    # ==========================================================

    def stream_projection_data(self, gc_data: Cluster_Data_initialization,
                               dt: u.Quantity = -0.5 * u.Myr, shift_rv=None,
                               num_steps=2800):
        # Get progenitor's initial phase-space position and mass
        progenitor_pos0 = gc_data.phase_space_position(skycoord_rv=shift_rv)
        progenitor_mass = gc_data.mass * u.Msun

        # Distinguish stream distribution
        stream_distribution_function = ms.ChenStreamDF()
        gc_inner_potential = gal_pot.PlummerPotential(m=progenitor_mass, b=4 * u.kpc,
                                                      units=galactic)

        # Generate mock stream
        stream_generator = ms.MockStreamGenerator(stream_distribution_function, self.H,
                                                  progenitor_potential=gc_inner_potential)
        stream_orbit, _ = stream_generator.run(progenitor_pos0, progenitor_mass,
                                               dt=dt, n_steps=num_steps)

        # Convert to observational coordinates
        stream_icrs = stream_orbit.to_coord_frame(coord.ICRS())
        stream_galactic = stream_orbit.to_coord_frame(coord.Galactic())

        # Define the coordinates
        coord_df = pd.DataFrame({
            "RA": stream_icrs.ra.degree,
            "Dec": stream_icrs.dec.degree,
            "X": stream_orbit.x,
            "Y": stream_orbit.y,
            "Z": stream_orbit.z,
            "RV": stream_icrs.radial_velocity.to(u.km / u.s).value,
            "pm_ra": stream_icrs.pm_ra_cosdec.to(u.mas / u.yr).value,
            "pm_dec": stream_icrs.pm_dec.to(u.mas / u.yr).value,
            "l": stream_galactic.l,
            "b": stream_galactic.b,
            "Release Time": stream_orbit.release_time.value
        })
        #print(stream_galactic, stream_icrs)
        return stream_orbit, coord_df, stream_galactic, stream_icrs, progenitor_pos0

    # ==========================================================
    # ORBITAL PROJECTION
    # ==========================================================
    def Orbit_display(self, gc_data: Cluster_Data_initialization,
                       shift_rv=None, dt: u.Quantity = -0.5*u.Myr,
                      color=None, make_2dplot: bool = True, make_3dplot: bool = True):
        # ==========================================================
        # ORBIT GRAPH INITIALIZATION
        # ==========================================================
        progenitor_pos0 = gc_data.phase_space_position(skycoord_rv=shift_rv)

        # Debugging
        # print(f"Type of progenitor_pos0: {type(progenitor_pos0)}")
        # print(f"Shape of position: {progenitor_pos0.pos.shape}")
        # print(f"Shape of velocity: {progenitor_pos0.vel.shape}")
        # print(f"Position values: {progenitor_pos0.pos}")
        # print(f"Velocity values: {progenitor_pos0.vel}")
        # Distinguish between 2d and 3d graphs
        orbit_2dfigure = None
        orbit_3dfigure = None

        # ==========================================================
        # 2D ORBIT GRAPH
        # ==========================================================
        # Integrate the orbit through the Hamiltonian Potential
        num_steps = 500
        orbit_rotating = self.H.integrate_orbit(progenitor_pos0, dt=dt, n_steps=num_steps)


        # Configure back to static framing for graph
        orbit_inertial = orbit_rotating.to_frame(gal_pot.StaticFrame(galactic))

        if make_2dplot:
            # Set graphing plot
            orbit_2dfigure = orbit_inertial.plot(color=color)

        # ==========================================================
        # 3D ORBIT GRAPH
        # ==========================================================
            # Integrate the orbit through the Hamiltonian Potential
            # num_steps = 500
            # orbit_rotating = self.H.integrate_orbit(progenitor_pos0, dt=dt, n_steps=num_steps)
            #
            # # Configure back to static framing for graph
            # orbit_inertial = orbit_rotating.to_frame(gal_pot.StaticFrame(galactic))
        if make_3dplot:
            # Set graph figure and axes
            orbit_3dfigure = plt.figure(figsize = (6,6))
            ax = orbit_3dfigure.add_subplot(111, projection='3d')

            # Plot the orbit
            ax.plot(orbit_inertial.x.value, orbit_inertial.y.value, orbit_inertial.z.value,
                    color= color, marker = 'o', markersize = 0.5, linestyle = '-', linewidth= '0.5')

            # Set labels
            ax.set_xlabel('X [kpc]')
            ax.set_ylabel('Y [kpc]')
            ax.set_zlabel('Z [kpc]')

        return orbit_2dfigure, orbit_3dfigure



    # ==========================================================
    # RV VS RA STREAM GRAPH
    # ==========================================================
    def generate_ra_rv_stream(self, gc_data: Cluster_Data_initialization,
                          stream_icrs: coord.ICRS):
        radial_velocity_plot_fig, radial_velocity_plot_axes = plt.subplots(1,1, figsize = (10,5), sharex=True)

        # Plot stream particles
        radial_velocity_plot_axes.scatter(stream_icrs.ra.degree,
                                          stream_icrs.radial_velocity.to(u.km/u.s),
                                          marker = '.', s=2, alpha = 0.95, color = 'k')
        # Plot cluster position
        cluster_ra = gc_data.ra
        cluster_rv = gc_data.rv

        radial_velocity_plot_axes.scatter([cluster_ra], [cluster_rv],
                                          marker = '*', s=100, alpha = 0.95, color = 'k')
        radial_velocity_plot_axes.scatter([cluster_ra], [cluster_rv],
                                          marker='*', s=50, alpha=0.5, color='g')

        # Reference lines at cluster position
        radial_velocity_plot_axes.axvline(x = cluster_ra, alpha = 0.5, color = 'g')
        radial_velocity_plot_axes.axhline(y = cluster_rv, alpha = 0.5, color = 'g')

        # Labels and title
        plt.title("Radial Velocity vs Right Ascension", fontsize = 20)
        radial_velocity_plot_axes.set_xlabel(r'$\alpha\, [{\rm deg}]$', fontsize = 20)
        radial_velocity_plot_axes.set_ylabel(r'$v_r\, [{\rm km}\, {\rm s}^{-1}]$', fontsize = 20)

        return radial_velocity_plot_fig, radial_velocity_plot_axes

    # ==========================================================
    # RV VS DEC STREAM GRAPH
    # ==========================================================
    def generate_ra_dec_stream(self, gc_data: Cluster_Data_initialization,
                           stream_icrs: coord.ICRS):

        ra_dec_plot_fig, ra_dec_plot_axes = plt.subplots(1, 1, figsize=(10, 5), sharex=True)

        # Plot the stream
        ra_dec_plot_axes.scatter(stream_icrs.ra.degree, stream_icrs.dec.degree,
                                 marker='.', s=5, alpha=0.25, color='g')

        # Pal 5 cluster coordinates
        cluster_ra = gc_data.ra
        cluster_dec = gc_data.dec

        # Plot cluster position with multiple marker sizes
        ra_dec_plot_axes.scatter([cluster_ra], [cluster_dec],
                                 marker='*', s=100, alpha=0.95, color='k')
        ra_dec_plot_axes.scatter([cluster_ra], [cluster_dec],
                                 marker='*', s=50, alpha=0.5, color='g')

        # Reference lines at cluster position
        ra_dec_plot_axes.axvline(x=cluster_ra, alpha=0.5, color='g')
        ra_dec_plot_axes.axhline(y=cluster_dec, alpha=0.5, color='g')

        # Labels
        plt.title("Radial Velocity vs Declination", fontsize=20)
        ra_dec_plot_axes.set_xlabel(r'$\alpha\,[{\rm deg}]$', fontsize=20)
        ra_dec_plot_axes.set_ylabel(r'$Dec\,[{\rm deg}]$', fontsize=20)

        return ra_dec_plot_fig, ra_dec_plot_axes

    # ==========================================================
    # INTERACTIVE 3D PLOT
    # ==========================================================

    def stream_plot_3d (self, gc_data: Cluster_Data_initialization,
                        stream_orbit):

        # Create interactive 3d plot
        tidal_tails_fig = px.scatter_3d(
            x = stream_orbit.x,
            y = stream_orbit.y,
            z = stream_orbit.z,
            labels={'x': 'X [kpc]', 'y': 'Y [kpc]', 'z': 'Z [kpc]'},
            opacity=.6
        )
        # Set size of star markers
        tidal_tails_fig.update_traces(
            marker = dict(size = 1 ) # Update as needed

        )
        # Set size of the plot itself
        tidal_tails_fig.update_layout(
            width = 800,
            height = 800,
            scene = dict(
                aspectmode = 'cube'
            )
        )
        return tidal_tails_fig









