#!/bin/bash
cp /workspace/output/malayalam_vits-March-02-2026_06+03AM-224064f/best_model.pth /workspace/best_model_backup.pth
git add /workspace/best_model_backup.pth
git commit -m "Auto backup best model"
git push

