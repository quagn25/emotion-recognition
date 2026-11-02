import tensorflow as tf
print("Số lượng GPU khả dụng: ", len(tf.config.list_physical_devices('GPU')))

