from unittest import TestCase

import pylangkit.naive_bayes as nb


class TestNaiveBayes(TestCase):

    def setUp(self):
        """Initialize data set for tests."""
        self.train_data = [('Chinese Beijing Chinese', '0'),
                           ('Chinese Chinese Shanghai', '0'),
                           ('Chinese Macao', '0'),
                           ('Tokyo Japan Chinese', '1')]

    def test_trained_model(self):
        """Test trained model."""
        # Create and train model
        model = nb.NaiveBayes()
        model.fit(self.train_data, frac=1.)

        # Get predictions
        predicted = model.classify(['Chinese Chinese Chinese Tokyo Japan'])

        self.assertTrue(predicted[0][1] == '0')

    def test_save_model_success(self):
        """Test save trained model successfully."""
        # Create, train, and save model
        model = nb.NaiveBayes()
        model.fit(self.train_data, frac=1.)
        model.dump_model('resources/test_model.pickle')
        del model

        # Create and load trained model from file
        other_model = nb.NaiveBayes()
        other_model.load_model('resources/test_model.pickle')

        # Get predictions
        predicted = other_model.classify(['Chinese Chinese Chinese Tokyo Japan'])

        self.assertTrue(predicted[0][1] == '0')

    def test_save_model_fail_no_file(self):
        """Test save trained model unsuccessfully."""
        self.assertRaises(IOError, nb.NaiveBayes().dump_model, '/fake')

    def test_load_model_success(self):
        """Test load trained model successfully."""
        # Create and load trained model from file
        model = nb.NaiveBayes()
        model.load_model('resources/test_model.pickle')

        # Get predictions
        predicted = model.classify(['Chinese Chinese Chinese Tokyo Japan'])

        self.assertTrue(predicted[0][1] == '0')

    def test_load_model_file_not_found(self):
        """Test save trained model unsuccessfully."""
        self.assertRaises(IOError, nb.NaiveBayes().load_model, '/fake')
