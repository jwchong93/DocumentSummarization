import modelCreation as mc
import dataWriter as dw
class trainer:
    def __init__(self):
        self.creator = mc.modelCreation()
        self.writer = dw.dataWriter()
        self.model = None
        self.dataManager = None
        self.batch_size = 32  # Batch size for training.
        self.epochs = 100  # Number of epochs to train for.
        pass

    def sequenceToSequenceTrain (self):
        self.model, self.dataManager = self.creator.sequenceToSequenceModel()
        self.model.fit([self.dataManager.inputData, self.dataManager.outputData], self.dataManager.targetData,
                       batch_size=self.batch_size, epochs=self.epochs, validation_split=0.2)
        # Save model
        self.creator.saveCurrentModelToFile(self.model)
        self.writer.writeProgress(self.creator.PROGRESS_PATH,
                                  self.creator.current_progress + self.creator.NUMBER_OF_SAMPLE)
