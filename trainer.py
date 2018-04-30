import modelCreation as mc
import dataWriter as dw
class trainer:
    def __init__(self):
        self.creator = mc.modelCreation()
        self.writer = dw.dataWriter()
        self.model = None
        self.dataManager = None
        self.batch_size = 64  # Batch size for training.
        self.epochs = 50  # Number of epochs to train for.
        self.iteration = 16
        pass

    def sequenceToSequenceTrain (self):
        self.model, self.dataManager = self.creator.sequenceToSequenceModel()

        for i in range(self.iteration):
            self.model.fit([self.dataManager.inputData, self.dataManager.outputData], self.dataManager.targetData,
                           batch_size=self.batch_size, epochs=self.epochs, validation_split=0.2)
            # Save model
            self.writer.writeProgress(self.creator.PROGRESS_PATH,
                                      self.creator.current_progress + self.creator.NUMBER_OF_SAMPLE)
            self.creator.saveCurrentModelToFile(self.model)
            self.creator.createTokenizerFromTrainingData(self.creator.TRAINING_DATA_PATH, self.creator.PROGRESS_PATH)
