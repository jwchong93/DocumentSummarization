import modelCreation as mc
import dataWriter as dw
import matplotlib.pyplot as plt

class trainer:
    def __init__(self):
        self.creator = mc.modelCreation()
        self.writer = dw.dataWriter()
        self.model = None
        self.dataManager = None
        self.batch_size = 64  # Batch size for training.
        self.epochs = 300  # Number of epochs to train for.
        self.iteration = 16
        pass

    def sequenceToSequenceTrain (self):
        self.model, self.dataManager = self.creator.sequenceToSequenceModelTrain()
        scores = self.model.evaluate([self.dataManager.inputData, self.dataManager.outputData], self.dataManager.targetData,
                            batch_size=self.batch_size)
        print(scores)
        for i in range(self.iteration):

            history = self.model.fit([self.dataManager.inputData, self.dataManager.outputData], self.dataManager.targetData,
                           batch_size=self.batch_size, epochs=self.epochs, validation_split=0.2)
            plt.plot(history.history['acc'])
            plt.plot(history.history['val_acc'])
            plt.title('training accuracy')
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper left')
            plt.savefig("Accuracy_" + str(i+1))
            plt.clf()

            # summarize history for loss
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('training loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper left')
            plt.savefig("Loss_" + str(i + 1))
            plt.clf()
            # Save model
            self.writer.writeProgress(self.creator.PROGRESS_PATH,
                                      self.creator.current_progress + self.creator.NUMBER_OF_SAMPLE)
            self.creator.saveCurrentModelToFile(self.model)
            self.creator.refreshData()

        scores = self.model.evaluate([self.dataManager.inputData, self.dataManager.outputData],
                                     self.dataManager.targetData,
                                     batch_size=self.batch_size)
        print(scores)

    def sequenceToSequenceInference(self):
        self.creator.sequenceToSequenceModelInference()