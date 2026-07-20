# Upload instructions

This directory is intended to be pushed as a standalone repository containing only
the SceneDiffuser++ SUMO/Nanjing reproduction code.

Expected remote:

```powershell
https://github.com/TJINTO/SceneDiffuser++.git
```

Before pushing, create an empty GitHub repository named `SceneDiffuser++` under
`TJINTO` and do not initialize it with a README, license, or `.gitignore`.

Then run:

```powershell
cd D:\Python_Projects\SceneDiffuser++
git remote add origin https://github.com/TJINTO/SceneDiffuser++.git
git push -u origin main
```

If GitHub does not accept `++` in the repository name, use
`SceneDiffuserPlusPlus` or `SceneDiffuserPP` and set the remote URL accordingly.
