
import numpy as np
from util import  load_and_preprocess_test_images,sp_and_re_test


TEST_DI = 'Data/testscriptdata'
IMG_SIZ=100
test_sample=load_and_preprocess_test_images(TEST_DI,IMG_SIZ)
x_test,y_test= sp_and_re_test(test_sample)
model_path = 'svm_model.npy'
loaded_model = np.load(model_path, allow_pickle=True).item()

predictions = loaded_model.predict(x_test.reshape(len(x_test), -1))
accuracy = np.mean(predictions == np.argmax(y_test, axis=1))
print(f"Accuracy of svm for testing : {accuracy * 100}")




