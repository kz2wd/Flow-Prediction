#!/bin/bash
rsync -avz --delete --exclude='.git/' --exclude='__pycache__/' --include='FolderManager.py' --include='main.py' --include='space_exploration/***' --include='visualization/***' --exclude='*' ./ lab:~/code/sync
