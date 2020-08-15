# Contributor Guide


The purpose of this guide is to provide the necessary background
such that you can make improvements to the OG-USA and share them with
others working on the model.

OG-USA code is tracked by using the version control software
[`Git`](https://help.github.com/articles/github-glossary/#git
).  Following the next steps will get you up and running and
contributing to the model even if you've never used Git or other
version control software.

If you have already completed the Setup Python and Setup Git sections,
please skip to [Workflow](#workflow_link).

## Setup Python


OG-USA is written in the Python programming language.
Download and install the free Anaconda distribution of Python 3.6 from
[Anaconda](https://www.anaconda.com/download/).  You must do this even if you already have
Python installed on your computer because the Anaconda distribution
contains additional Python packages that are used by OG-USA (many of
which are not included in other Python installations).  You can install
the Anaconda distribution without having administrative privileges on your
computer and the Anaconda distribution will not interfere with any Python
installation that came as part of your computer's operating system.

## Setup Git

1. Create a [GitHub](https://github.com/) user account.

2. Install Git on your local machine by following steps 1-4 on [Git
   setup](https://help.github.com/articles/set-up-git/).

3. Tell Git to remember your GitHub password by following steps 1-4 on
   [password setup](https://help.github.com/articles/caching-your-github-password-in-git/).

4. Sign in to GitHub and create your own [remote](https://help.github.com/articles/github-glossary/#remote) [repository](https://help.github.com/articles/github-glossary/#repository)
   (repo) of OG-USA by clicking [Fork](https://help.github.com/articles/github-glossary/#fork) in the upper
   right corner of the [OG-USA GitHub page](https://github.com/PSLmodels/OG-USA). Select your
   username when asked "Where should we fork this repository?"

5. From your command line, navigate to the directory on your computer
   where you would like your local repo to live.

6. Create a local repo by entering at the command line the text after
   the $.[^commandline_note]  This step creates a directory called OG-USA in
   the directory that you specified in the prior step:

    ```
       git clone https://github.com/[github-username]/OG-USA.git
    ```

7. From your command line or terminal, navigate to your local
   OG-USA directory.

8. Make it easier to [push](https://help.github.com/articles/github-glossary/#pull) your local work to others and [pull](https://help.github.com/articles/github-glossary/#pull)
   others' work to your local machine by entering at the command line::
    ```
      $ cd OG-USA
      OG-USA$ git remote add upstream https://github.com/open-source-economics/OG-USA.git
    ```

9. Create a conda environment with all of the necessary packages to
   execute the source code:

    ```
      OG-USA$ conda env create
    ```

10. The prior command will create a conda environment called `ogusa-dev`.
    Activate this environment as follows:
    ```
      OG-USA$ source activate ogusa-dev
    ```
    If you are working on Windows, use the following from the command line::
    ```
      OG-USAr$ activate ogusa-dev
    ```

If you've made it this far, you've successfully made a remote copy (a
fork) of the central OG-USA repo. That remote repo is hosted on
GitHub.com. You've also created a local repo (a [clone](https://help.github.com/articles/github-glossary/#clone)) that lives
on your machine and only you can see; you will make your changes to
the OG-USA model by editing the files in the OG-USA
directory on your machine and then submitting those changes to your
local repo. As a new contributor, you will push your changes from your
local repo to your remote repo when you're ready to share that work
with the team.

Don't be alarmed if the above paragraph is confusing. The following
section introduces some standard Git practices and guides you through
the contribution process.


## [Workflow](#workflow_link)

The following text describes a typical workflow for changing
OG-USA.  Different workflows may be necessary in some
situations, in which case other contributors are here to help.

1. Before you edit the OG-USA source code on your machine,
   make sure you have the latest version of the central OG-USA
   repository by executing the following **four** Git commands:

   a. Tell Git to switch to the master branch in your local repo.
      Navigate to your local OG-USA directory and enter the
      following text at the command line:

    ```
        OG-USA$ git checkout master
    ```

   b. Download all of the content from the central OG-USA repo:
    ```
        OG-USA$ git fetch upstream
    ```
   c. Update your local master branch to contain the latest content of
      the central master branch using [merge](https://help.github.com/articles/github-glossary/#merge). This step ensures that
      you are working with the latest version of OG-USA:
    ```
        OG-USA$ git merge upstream/master
    ```
   d. Push the updated master branch in your local repo to your GitHub repo:
    ```
        OG-USA$ git push origin master
    ```
2. Create a new [branch](https://help.github.com/articles/github-glossary/#branch) on your local machine. Think of your
   branches as a way to organize your projects. If you want to work on
   this documentation, for example, create a separate branch for that
   work. If you want to change an element of the OG-USA model, create
   a different branch for that project:
    ```
     OG-USA$ git checkout -b [new-branch-name]
    ```
3. As you make changes, frequently check that your changes do not
   introduce bugs or degrade the accuracy of the OG-USA. To do
   this, run the following command from the command line from inside
   the OG-USA/ogusa directory:
    ```
     OG-USA/ogusa$ py.test
    ```
   If the tests do not pass, try to fix the issue by using the
   information provided by the error message. If this isn't possible
   or doesn't work, we are here to help.

4. Now you're ready to [commit](https://help.github.com/articles/github-glossary/#commit) your changes to your local repo using
   the code below. The first line of code tells Git to track a
   file. Use "git status" to find all the files you've edited, and
   "git add" each of the files that you'd like Git to track. As a
   rule, do not add large files. If you'd like to add a file that is
   larger than 25 MB, please contact the other contributors and ask how to
   proceed. The second line of code commits your changes to your local
   repo and allows you to create a commit message; this should be a
   short description of your changes.

   *Tip*: Committing often is a good idea as Git keeps a record of
   your changes. This means that you can always revert to a previous
   version of your work if you need to.  Do this to commit:
    ```
     OG-USA$ git add [filename]
     OG-USA$ git commit -m "[description-of-your-commit]"
    ```
5. Periodically, make sure that the branch you created in step 2
   is in sync with the changes other contributors are making to
   the central master branch by fetching upstream and merging
   upstream/master into your branch::
    ```
      OG-USA$ git fetch upstream
      OG-USA$ git merge upstream/master
    ```
   You may need to resolve conflicts that arise when another
   contributor changed the same section of code that you are
   changing. Feel free to ask other contributors for guidance
   if this happens to you. If you do need to fix a merge
   conflict, re-run the test suite afterwards (step 4.)

6. When you are ready for other team members to review your code, make
   your final commit and push your local branch to your remote repo:
    ```
     OG-USA$ git push origin [new-branch-name]
    ```
7. From the GitHub.com user interface, [open a pull request](https://help.github.com/articles/creating-a-pull-request/#creating-the-pull-request).

8. When you open a GitHub pull request, a code coverage report will be
   automatically generated.  If your branch adds new code that is not
   tested, the code coverage percent will decline and the number of
   untested statements ("misses" in the report) will increase.  If
   this happens, you need to add to your branch one or more tests of
   your newly added code.  Add tests so that the number of untested
   statements is the same as it is on the master branch.


## Simple Usage


Examples coming soon...

[^commandline_note]:The dollar sign is the end of the command prompt on a Mac.  If
       you're on Windows, this is usually the right angle bracket (>).
       No matter the symbol, you don't need to type it (or anything to
       its left, which shows the current working directory) at the
       command line before you enter a command; the prompt symbol and
       preceding characters should already be there. 