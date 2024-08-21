import unittest
import json
from app import app


class FlaskAppTestCase(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_predict(self):
        response = self.app.post('/predict', data=json.dumps({
            'Age': 30,
            'Sex': 'male',
            'Job': 'skilled',
            'Housing': 'own',
            'Savingaccounts': 'little',
            'Checkingaccount': 'moderate',
            'Creditamount': 5000,
            'Duration': 24,
            'Purpose': 'car',
            'Saving_Checking_Account': 'little_moderate'
        }), content_type='application/json')
        
        self.assertEqual(response.status_code, 200)
        self.assertIn('risk', json.loads(response.data))

if __name__ == "__main__":
    unittest.main()
