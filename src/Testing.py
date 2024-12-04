import unittest
from itertools import product
from BlowFish_FacialRecognition import run

'''
resolutions = ['500x500', '1000x1000', '2000,2000']
extentions = ['png', 'jpg', 'tif', 'bmp']

enc_path_res = "500x500"
enc_path_ext = ".png"
dec_path_res = "500x500"
dec_path_ext = ".tif"
'''

class TestFiles(unittest.TestCase):

    
    def test_files(self):
        # Define the resolutions and extensions
        files = ['500x500.png','500x500.jpg','500x500.tif','500x500.bmp','1000x1000.png','1000x1000.jpg',
         '1000x1000.tif','1000x1000.bmp','2000x2000.png','2000x2000.jpg','2000x2000.tif','2000x2000.bmp']

        for enc in files:
            for dec in files:

                with self.subTest(enc_file=enc, dec_file=dec):
                    # Ensure the main function returns True for each combination
                    self.assertFalse(run(enc, dec))

if __name__ == '__main__':
    # Redirect test output to a log file
    with open('successful_combinations.log', 'w') as log_file:
        runner = unittest.TextTestRunner(stream=log_file, verbosity=2)
        unittest.main(testRunner=runner)

