@echo off
cd /d "%~dp0"

echo.
echo ===== Git Status =====
git status
echo.

echo ===== Adding files =====
git add .
echo Files added!
echo.

echo ===== Committing =====
git commit -m "docs: Add detailed README with project description

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
echo Commit done!
echo.

echo ===== Pushing to remote =====
git push origin main
echo.

echo ===== Push Complete =====
git log --oneline -5
echo.
pause
