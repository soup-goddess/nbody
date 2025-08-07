# A code that does step by step conversions of
# Equitorial to Galactic to Galactocentric cooridnates
# This code works for coordinates already realigned with epoch J2000.0 (ICRS)
# Please check coordinate metadata for confirmation of epoch J2000.0
# If equitorial coordinates are not realigned, simpmly take it's current position and subtract the objects proper motion multiplied by how many years since 2000
        # J2000.0 position = Current Position - (Proper motion * years since 2000)
import numpy as np
import astropy.units as u


class CoordinateConverter:

    # ==========================================================
    # INITIALIZE ALL PARAMETERS
    # ==========================================================
    def __init__(self, ra = None, dec = None, dis = None):
        self.ra = ra
        self.dec = dec
        self.dis = dis
        self.cartesian_coords = None

        # Solar parameters
        self.R_sun = 8.12 * u.kpc   # Distance from sun to center of Galaxy
        self.z_sun = 0.0208 * u.kpc   # Height of sun above galactic plane
        self.v_sun_pec = [11.1, 12.24, 7.25] * u.km/u.s     # Sun's peculiar velocity

        # Rotation parameters
        self.v_circ = 220 * u.km/u.s        # Circular velocity at the Sun's position (at J2000.0)

        # J2000.0 galactic coordinate constants
        self.delta_G = 27.128 * u.degree
        self.alpha_G = 192.859 * u.degree
        self.l_NCP = 122.932 * u.degree

    # ==========================================================
    # SET COORDINATES AND UNITS
    # ==========================================================
    def set_pos_coordinates(self, ra, dec, dis):
        self.ra = ra * u.degree
        self.dec = dec * u.degree
        self.dis = dis * u.kpc
        return self.ra, self.dec, self.dis

    # ==========================================================
    # CONVERT EQUITORIAL POSITION TO GALACTIC
    # ==========================================================
    def pos_equitorial_to_galactic(self):

        # Convert angular values to radians for calculations
        ra_rad = self.ra.to(u.radian).value
        dec_rad = self.dec.to(u.radian).value

        # Convert galactic pole coordinates to radians for calculations
        delta_rad = self.delta_G.to(u.radian).value
        alpha_rad = self.alpha_G.to(u.radian).value
        l_NCP_rad = self.l_NCP.to(u.radian).value


        # Begin conversion calculations
        # cos(l_NCP - l) * cos(b) = sin(δ) * cos(δ_G) - cos(δ) * sin(δ_G) * cos(α - α_G)
        gal_x = (np.sin(dec_rad) * np.cos(delta_rad) - np.cos(dec_rad)
                 * np.sin(delta_rad) * np.cos(ra_rad-alpha_rad))
        # sin(l_NCP - l) * cos(b) = cos(δ) * sin(α - α_G)
        gal_y = np.cos(dec_rad) * np.sin(ra_rad-alpha_rad)

        # sin(b) = sin(δ) * sin(δ_G) + cos(δ) * cos(δ_G) * cos(α - α_G)
        gal_z = (np.sin(dec_rad) * np.sin(delta_rad) + np.cos(dec_rad)
                 * np.cos(delta_rad) * np.cos(ra_rad-alpha_rad))

        # Calculate galactic latitude b
        b = np.arcsin(gal_z)

        # Calculate (l_NCP - 1) with arctan2 for proper quadrant placement and to avoid division by zero
        lNCP_minus_l = np.arctan2(gal_y, gal_x)

        # Calculate galactic longitude l
        l= l_NCP_rad - lNCP_minus_l

        # Convert back to degrees and normalize longitude
        l_deg = np.degrees(l) % 360
        b_deg = np.degrees(b)

        return l_deg * u.degree, b_deg * u.degree, self.dis

    # ===========================================================
    # CONVERT GALACTIC SPHERICAL POSITION TO GALACTIC CARTESIAN
    # ===========================================================
    def pos_galactic_spherical_to_cartesian(self, l, b, dis):

        # Convert l and b back to radians
        l_rad = l.to(u.radian).value
        b_rad = b.to(u.radian).value

        # Turn l and b into full cartesian coordinates
        x_gal = dis * np.cos(b_rad) * np.cos(l_rad)
        y_gal = dis * np.cos(b_rad) * np.sin(l_rad)
        z_gal = dis * np.sin(b_rad)

        return x_gal, y_gal, z_gal

    # ===========================================================
    # CONVERT CARTESIAN GALACTIC POSITION TO GALACTIC CENTRIC
    # ===========================================================
    def pos_galactic_to_galactocentric(self, x_gal, y_gal, z_gal):

        # Convert to numpy arrays for ease of handling, speed, and vectorized operations
        x_gal = np.array(x_gal)
        y_gal = np.array(y_gal)
        z_gal = np.array(z_gal)

        # Call solar position values
        R_sun_val = self.R_sun.to(u.kpc).value
        z_sun_val = self.z_sun.to(u.kpc).value

        # Convert coordinates
        # Sun is set at (-R_sun, 0, z_sun) in galactocentric coordinates
        x_gc = x_gal - R_sun_val        # Shift origin from heliocentric to galactic center
        y_gc = y_gal                    # No change in y direction as sun is set at 0
        z_gc = z_gal - z_sun_val        # Account for suns height above plane

        return x_gc * u.kpc, y_gc * u.kpc, z_gc * u.kpc

    # ==========================================================
    # CALCULATE DISTANCE FROM GALACTIC CENTER
    # ==========================================================
    def distance_from_galactic_center(self, x_gc, y_gc, z_gc):

        # Returns distance from coordinates to galactic center in kpc

        distance =  np.sqrt(x_gc**2 + y_gc**2 + z_gc**2)
        return distance * u.kpc
    # ==========================================================
    # CALCULATE GALACTOCENTRIC RADIUS
    # ==========================================================
    def galactic_radius(self, x_gc, y_gc):

        # Returns Galactocentric Radius of coordinate object in kpc
        radius =  np.sqrt(x_gc**2 + y_gc**2)
        return radius * u.kpc



    # =============================================================================
    # FULL COORDINATE CONVERSION (EQUITORIAL SPHERICAL TO CARTESIAN GALACTOCENTRIC)
    # =============================================================================
    def full_pos_conversion(self, ra, dec, distance):
        # Set coordinates
        self.set_coordinates(ra, dec, distance)

        # Step 1: Equatorial to galactic
        l_gal, b_gal, dist_gal = self.equitorial_to_galactic()

        # Step 2: Galactic spherical to cartesian
        x_gal, y_gal, z_gal = self.galactic_spherical_to_cartesian(l_gal, b_gal, dist_gal)

        # Step 3: Galactic to galactocentric
        x_gc, y_gc, z_gc = self.galactic_to_galactocentric(x_gal, y_gal, z_gal)

        # Additional calculations
        dist_gc = self.distance_from_galactic_center(x_gc, y_gc, z_gc)
        radius_gc = self.galactic_radius(x_gc, y_gc)

        return {
            'equitorial_spherical': {
                'ra_deg': ra,
                'dec_deg': dec,
                'distance_kpc': distance
            },
            'galactic_spherical': {
                'l_deg': l_gal.value,
                'b_deg': b_gal.value,
                'distance_kpc': dist_gal.value
            },
            'galactic_cartesian': {
                'x_gal_kpc': x_gal.value,
                'y_gal_kpc': y_gal.value,
                'z_gal_kpc': z_gal.value
            },
            'galactocentric': {
                'x_gc_kpc': x_gc.value,
                'y_gc_kpc': y_gc.value,
                'z_gc_kpc': z_gc.value
            },
            'derived_quantities': {
                'distance_from_gc_kpc': dist_gc.value,
                'galactocentric_radius_kpc': radius_gc.value
            }
        }

def main():
    print("Coordinate Converter")

    # Get Equitorial coordinate input
    ra = float(input("Enter Right Ascension in degrees: "))
    dec = float(input("Enter Declination in degrees: "))
    dis = float(input("Enter Distance in kpc: "))

    # Call conversion functions
    converter = CoordinateConverter()
    results = converter.full_pos_conversion(ra, dec, dis)

    # Show Galactic Centric coordinates
    x_gc = results['galactocentric']['x_gc_kpc']
    y_gc = results['galactocentric']['y_gc_kpc']
    z_gc = results['galactocentric']['z_gc_kpc']

    print(f"Galactocentric Coordinates: X = {x_gc}, Y = {y_gc}, Z = {z_gc} kpc")

if __name__ == "__main__":
    main()