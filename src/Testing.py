import unittest
from BlowFish_FacialRecognition import test
from BlowFish_FacialRecognition import unstable_run


'''
resolutions = ['500x500', '1000x1000', '2000,2000']
extentions = ['png', 'jpg', 'tif', 'bmp']

gender = ['M', 'F']
resolutions = ['512', '1024', '2048']
extentions = ['.png', '.tif', '.bmp', '.jpg']

f_files = ['F512.png','F512.tif','F512.bmp','F512.jpg','F1024.png','F1024.tif','F1024.bmp','F1024.jpg','F2048.png','F2048.tif','F2048.bmp','F2048.jpg']
m_files = ['M512.png','M512.tif','M512.bmp','M512.jpg','M1024.png','M1024.tif','M1024.bmp','M1024.jpg','M2048.png','M2048.tif','M2048.bmp','M2048.jpg']
'''



class TestFiles(unittest.TestCase):

    path = "..\\Faces\\"

    def test_M(self):
    # Define File and and extensions
        files = ['M512.png','M512.tif','M512.bmp','M512.jpg','M1024.png','M1024.tif','M1024.bmp','M1024.jpg','M2048.png','M2048.tif','M2048.bmp','M2048.jpg']
        print("M Testing")
        for enc in files:
            for dec in files:

                with self.subTest(enc_file=enc, dec_file=dec):
                    self.assertFalse(test(self.path+enc, self.path+dec))
    
    def test_F(self):
    # Define File and and extensions
        files = ['F512.png','F512.tif','F512.bmp','F512.jpg','F1024.png','F1024.tif','F1024.bmp','F1024.jpg','F2048.png','F2048.tif','F2048.bmp','F2048.jpg']
        print("F Testing")
        for enc in files:
            for dec in files:
                with self.subTest(enc_file=enc, dec_file=dec):
                    self.assertFalse(test(self.path+enc, self.path+dec))

    """def test_personal_imgs(self):
        # Define the resolutions and extensions
        files = ['500x500.png','500x500.jpg','500x500.tif','500x500.bmp','1000x1000.png','1000x1000.jpg',
         '1000x1000.tif','1000x1000.bmp','2000x2000.png','2000x2000.jpg','2000x2000.tif','2000x2000.bmp']

        for enc in files:
            for dec in files:

                with self.subTest(enc_file=enc, dec_file=dec):
                    self.assertTrue(run_comparison(enc, dec))
    """
    

if __name__ == '__main__':
    # Redirect test output to a log file
    with open('successful_combinations.log', 'w') as log_file:
        runner = unittest.TextTestRunner(stream=log_file, verbosity=2)
        unittest.main(testRunner=runner)

