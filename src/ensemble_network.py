import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import StackingClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score

class BaseModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BaseModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class EnsembleNetwork:
    def __init__(self, input_size, output_size):
        # Define multiple base models
        self.model1 = BaseModel(input_size, 128, output_size)
        self.model2 = BaseModel(input_size, 128, output_size)
        self.model3 = BaseModel(input_size, 128, output_size)

        # Optimizers for the models
        self.optimizer1 = optim.Adam(self.model1.parameters(), lr=1e-3)
        self.optimizer2 = optim.Adam(self.model2.parameters(), lr=1e-3)
        self.optimizer3 = optim.Adam(self.model3.parameters(), lr=1e-3)

        # Ensemble method (stacking with gradient boosting)
        self.ensemble_model = StackingClassifier(
            estimators=[
                ('model1', self.model1),
                ('model2', self.model2),
                ('model3', self.model3)
            ],
            final_estimator=GradientBoostingClassifier()
        )

    def train_base_model(self, model, optimizer, x, y):
        optimizer.zero_grad()
        output = model(x)
        loss = torch.nn.functional.mse_loss(output, y)
        loss.backward()
        optimizer.step()
        return loss.item()

    def train_ensemble(self, x, y):
        # Train individual models
        loss1 = self.train_base_model(self.model1, self.optimizer1, x, y)
        loss2 = self.train_base_model(self.model2, self.optimizer2, x, y)
        loss3 = self.train_base_model(self.model3, self.optimizer3, x, y)

        # Train the ensemble model (meta-learner)
        self.ensemble_model.fit(x.detach().cpu().numpy(), y.detach().cpu().numpy())

        return (loss1 + loss2 + loss3) / 3

    def predict(self, x):
        # Get predictions from the ensemble model
        return self.ensemble_model.predict(x.detach().cpu().numpy())

    def cross_validate(self, x, y, cv=5):
        # Perform cross-validation to evaluate ensemble model performance
        scores = cross_val_score(self.ensemble_model, x.detach().cpu().numpy(), y.detach().cpu().numpy(), cv=cv)
        return scores.mean()

    def save_model(self, path="ensemble_model.pth"):
        torch.save({
            'model1': self.model1.state_dict(),
            'model2': self.model2.state_dict(),
            'model3': self.model3.state_dict(),
            'ensemble_model': self.ensemble_model
        }, path)

    def load_model(self, path="ensemble_model.pth"):
        checkpoint = torch.load(path)
        self.model1.load_state_dict(checkpoint['model1'])
        self.model2.load_state_dict(checkpoint['model2'])
        self.model3.load_state_dict(checkpoint['model3'])
        self.ensemble_model = checkpoint['ensemble_model']