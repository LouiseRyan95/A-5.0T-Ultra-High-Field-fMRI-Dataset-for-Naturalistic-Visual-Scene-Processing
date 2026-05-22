function nvd_naturalistic_visual_scene_task(varargin)
%NVD_NATURALISTIC_VISUAL_SCENE_TASK Run the naturalistic visual scene task.
%
% This Psychtoolbox script presents natural images with interleaved blank
% trials and a central fixation dot. Blank trials can optionally trigger a
% fixation-color change. The script was prepared for public release with
% English comments.
%
% Task structure
% --------------
% - Image trials: a natural image is shown for 1 s.
% - Blank trials: a white rectangle is shown for 1 s.
% - Inter-stimulus interval: gray background for 2--3 s.
% - Fixation: a red/blue central dot is shown throughout the task.
% - Responses: numeric keys 1--4 are monitored and logged.
%
% Required files
% --------------
% - A directory containing stimulus images, for example COCO images.
%
% Example
% -------
% nvd_naturalistic_visual_scene_task('subject','001', ...
%     'session','101', ...
%     'imageFolder','/path/to/stimuli/val2017', ...
%     'outputDir','/path/to/task_logs', ...
%     'debug',false);
%
% Notes
% -----
% - Use debug=false and syncCheck=true for scanner acquisition.
% - Edit stimulus and display parameters below only if they match the
%   acquisition protocol.
% - Press ESC to abort.

%% Parse input arguments.
p = inputParser;
p.addParameter('subject', '00', @(x) ischar(x) || isstring(x));
p.addParameter('session', '01', @(x) ischar(x) || isstring(x));
p.addParameter('imageFolder', './val2017', @(x) ischar(x) || isstring(x));
p.addParameter('imagePattern', '*.jpg', @(x) ischar(x) || isstring(x));
p.addParameter('outputDir', './logs', @(x) ischar(x) || isstring(x));
p.addParameter('debug', true, @(x) islogical(x) || isnumeric(x));
p.addParameter('syncCheck', true, @(x) islogical(x) || isnumeric(x));
p.addParameter('verbose', true, @(x) islogical(x) || isnumeric(x));
p.parse(varargin{:});

subject_id = char(p.Results.subject);
session_id = char(p.Results.session);
imageFolder = char(p.Results.imageFolder);
imagePattern = char(p.Results.imagePattern);
outputDir = char(p.Results.outputDir);
debug = logical(p.Results.debug);
syncCheck = logical(p.Results.syncCheck);
verbose = logical(p.Results.verbose);

%% Experiment parameters.
fontSize = 50;
fixDotDiameterPix = 12;

% Blank-trial settings.
nBlanks = 15;
minIntervalBetweenBlanks = 5;
colorChangeProb = 0.5;

% Timing parameters in seconds.
stimDuration = 1.0;
isiMin = 2.0;
isiMax = 3.0;

% Window settings for debug mode.
debugWindowLocation = [200 200];
debugWindowSize = [1920 1080];

% Font and colors.
textFont = 'Simsun';
fontColor = [0 0 0];
blankColor = [1 1 1];
backgroundColor = [0.5 0.5 0.5];
fixColorList = {[255 0 0], [0 0 255]};
fixColorIndex = 1;

% Maximum stimulus drawing size in pixels.
maxImageWidth = 800;
maxImageHeight = 600;

%% Load stimuli and generate the trial sequence.
imageFiles = dir(fullfile(imageFolder, imagePattern));
if isempty(imageFiles)
    error('No stimulus images matching %s were found in %s.', imagePattern, imageFolder);
end

allImageNames = {imageFiles.name};
shuffledImages = allImageNames(randperm(numel(allImageNames)));
imagePtr = 1;

nImageTrials = numel(allImageNames);
nTrials = nImageTrials + nBlanks;
trialSequence = zeros(1, nTrials);

% Trial codes: 0=image, 1=blank without fixation-color change, 2=blank with
% fixation-color change.
if nBlanks > 0
    blankIndices = generateBlankIndicesWithMinInterval(nTrials, nBlanks, minIntervalBetweenBlanks);
    nActualBlanks = numel(blankIndices);

    exactColorChanges = nActualBlanks * colorChangeProb;
    if exactColorChanges == floor(exactColorChanges)
        nColorChanges = exactColorChanges;
    elseif rand() < 0.5
        nColorChanges = ceil(exactColorChanges);
    else
        nColorChanges = floor(exactColorChanges);
    end

    colorChangeFlags = [true(1, nColorChanges), false(1, nActualBlanks - nColorChanges)];
    colorChangeFlags = colorChangeFlags(randperm(nActualBlanks));
    trialSequence(blankIndices) = 1 + colorChangeFlags;
end

%% Prepare output files.
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

timeString = char(datetime('now', 'Format', 'yyyyMMdd-HHmmss'));
baseName = sprintf('sub-%s_ses-%s_task-nvd_%s', subject_id, session_id, timeString);
logFilePath = fullfile(outputDir, [baseName '.log']);
eventsFilePath = fullfile(outputDir, [baseName '_events.tsv']);

logFile = fopen(logFilePath, 'w');
if logFile == -1
    error('Could not create log file: %s', logFilePath);
end

% Trial-level timing table. This table is saved in a BIDS-like events.tsv
% format at the end of the task or after an early abort.
eventRows = table('Size', [0 8], ...
    'VariableTypes', {'double','double','double','string','string','double','string','double'}, ...
    'VariableNames', {'trial_index','onset','duration','trial_type','stimulus_file','response_time','response_key','fixation_color_changed'});

%% Run the task.
try
    logMessage(logFile, verbose, '=== NVD naturalistic visual scene task started ===');
    logMessage(logFile, verbose, sprintf('Subject: %s; session: %s', subject_id, session_id));
    logMessage(logFile, verbose, sprintf('Stimuli: %d images from %s', nImageTrials, imageFolder));
    logMessage(logFile, verbose, sprintf('Total trials: %d; blank trials: %d', nTrials, nBlanks));

    AssertOpenGL;
    PsychDefaultSetup(2);
    KbName('UnifyKeyNames');

    if ~syncCheck
        Screen('Preference', 'SkipSyncTests', 1);
        logMessage(logFile, verbose, 'Synchronization tests were skipped. Use only for debugging.');
    end

    Screen('Preference', 'VisualDebugLevel', 0);
    Screen('Preference', 'TextEncodingLocale', 'UTF8');
    rng('shuffle');

    screens = Screen('Screens');
    screenNumber = max(screens);

    if debug
        windowRect = [debugWindowLocation, debugWindowLocation + debugWindowSize];
        [win, winRect] = PsychImaging('OpenWindow', screenNumber, backgroundColor, windowRect, 32, 2);
    else
        [win, winRect] = PsychImaging('OpenWindow', screenNumber, backgroundColor, [], 32, 2);
        HideCursor;
    end

    Screen('TextFont', win, textFont);
    Screen('TextSize', win, fontSize);
    Screen('TextStyle', win, 1);

    [xCenter, yCenter] = RectCenter(winRect);
    ifi = Screen('GetFlipInterval', win);

    startMessage = sprintf(['Press S to start\n\n' ...
        'Total trials: %d\n' ...
        'Blank trials: %d\n' ...
        'Stimulus images: %d\n\n' ...
        'Press ESC to abort'], nTrials, nBlanks, nImageTrials);
    Screen('FillRect', win, backgroundColor);
    DrawFormattedText(win, double(startMessage), 'center', 'center', fontColor);
    Screen('Flip', win);

    logMessage(logFile, verbose, 'Waiting for S key to start.');
    waitForStartOrAbort(logFile, verbose);

    Screen('FillRect', win, backgroundColor);
    drawFix(win, fixDotDiameterPix, fixColorList{fixColorIndex}, [xCenter yCenter]);
    vbl = Screen('Flip', win);
    taskStartTime = vbl;
    logMessage(logFile, verbose, sprintf('Task start GetSecs timestamp: %.6f', taskStartTime));

    for trialIndex = 1:nTrials
        if checkAbort()
            logMessage(logFile, verbose, 'ESC detected. The task was aborted by the operator.');
            break;
        end

        trialCode = trialSequence(trialIndex);
        stimulusFile = "n/a";
        trialType = "image";
        fixationColorChanged = 0;

        KbQueueCreate;
        KbQueueStart;

        if trialCode == 0
            if imagePtr > numel(shuffledImages)
                shuffledImages = allImageNames(randperm(numel(allImageNames)));
                imagePtr = 1;
                logMessage(logFile, verbose, 'Image list exhausted; reshuffled image order.');
            end

            imgName = shuffledImages{imagePtr};
            imagePtr = imagePtr + 1;
            stimulusFile = string(imgName);
            imgPath = fullfile(imageFolder, imgName);

            tex = makeImageTexture(win, imgPath);
            destRect = computeDestRectFromTex(tex, winRect, maxImageWidth, maxImageHeight);
            Screen('DrawTexture', win, tex, [], destRect, 0);
            drawFix(win, fixDotDiameterPix, fixColorList{fixColorIndex}, [xCenter yCenter]);
            stimulusOnset = Screen('Flip', win, vbl + 0.5 * ifi);
            vbl = stimulusOnset;
            logMessage(logFile, verbose, sprintf('Trial %03d image onset %.6f: %s', trialIndex, stimulusOnset, imgName));
            WaitSecs('UntilTime', stimulusOnset + stimDuration);
            Screen('Close', tex);

        elseif trialCode == 1 || trialCode == 2
            trialType = "blank";
            if trialCode == 2
                fixColorIndex = 3 - fixColorIndex;
                fixationColorChanged = 1;
            end

            blankRect = computeBlankRect(winRect, maxImageWidth, maxImageHeight);
            Screen('FillRect', win, backgroundColor);
            Screen('FillRect', win, blankColor, blankRect);
            drawFix(win, fixDotDiameterPix, fixColorList{fixColorIndex}, [xCenter yCenter]);
            stimulusOnset = Screen('Flip', win, vbl + 0.5 * ifi);
            vbl = stimulusOnset;
            logMessage(logFile, verbose, sprintf('Trial %03d blank onset %.6f; fixation color changed=%d', ...
                trialIndex, stimulusOnset, fixationColorChanged));
            WaitSecs('UntilTime', stimulusOnset + stimDuration);
        else
            error('Unknown trial code at trial %d: %d', trialIndex, trialCode);
        end

        [responseKey, responseTime] = collectResponse(stimulusOnset);
        KbQueueRelease;

        eventRows = [eventRows; {trialIndex, stimulusOnset - taskStartTime, stimDuration, ...
            trialType, stimulusFile, responseTime, responseKey, fixationColorChanged}]; %#ok<AGROW>

        isi = isiMin + rand() * (isiMax - isiMin);
        Screen('FillRect', win, backgroundColor);
        drawFix(win, fixDotDiameterPix, fixColorList{fixColorIndex}, [xCenter yCenter]);
        vbl = Screen('Flip', win, vbl + 0.5 * ifi);
        WaitSecs('UntilTime', vbl + isi);
    end

    writetable(eventRows, eventsFilePath, 'FileType', 'text', 'Delimiter', '\t');
    logMessage(logFile, verbose, sprintf('Saved event log: %s', eventsFilePath));

    Screen('FillRect', win, backgroundColor);
    DrawFormattedText(win, double('End of session'), 'center', 'center', fontColor);
    endFlip = Screen('Flip', win);
    WaitSecs('UntilTime', endFlip + 3);

    cleanupTask(logFile, debug);

catch ME
    if exist('eventRows', 'var') && height(eventRows) > 0
        try
            writetable(eventRows, eventsFilePath, 'FileType', 'text', 'Delimiter', '\t');
        catch
        end
    end

    if exist('logFile', 'var') && logFile ~= -1
        logMessage(logFile, true, '*** Error occurred ***');
        logMessage(logFile, true, sprintf('Identifier: %s', ME.identifier));
        logMessage(logFile, true, sprintf('Message: %s', ME.message));
        for iStack = 1:numel(ME.stack)
            logMessage(logFile, true, sprintf('Stack %d: %s, line %d', iStack, ME.stack(iStack).name, ME.stack(iStack).line));
        end
    end

    cleanupTask(logFile, debug);
    psychrethrow(psychlasterror);
end
end

function waitForStartOrAbort(logFile, verbose)
% Wait until S is pressed, or abort if ESC is pressed.
while true
    [secs, keyCode] = KbPressWait;
    if keyCode(KbName('s')) || keyCode(KbName('S'))
        logMessage(logFile, verbose, sprintf('S key detected at %.6f.', secs));
        break;
    elseif keyCode(KbName('ESCAPE'))
        error('NVDTask:UserAbort', 'ESC was pressed before task start.');
    end
end
end

function drawFix(win, dotDiamPix, color, centerXY)
% Draw a circular fixation dot.
pos = [centerXY(1); centerXY(2)];
Screen('DrawDots', win, pos, dotDiamPix, color, [], 2);
end

function tex = makeImageTexture(win, imgPath)
% Load an image and convert it to a Psychtoolbox texture.
img = imread(imgPath);
if ~isa(img, 'uint8')
    img = im2uint8(img);
end
if ndims(img) == 2
    img = repmat(img, [1 1 3]);
end
tex = Screen('MakeTexture', win, img);
end

function destRect = computeDestRectFromTex(tex, winRect, maxWidth, maxHeight)
% Compute a centered destination rectangle while preserving aspect ratio.
winW = RectWidth(winRect);
winH = RectHeight(winRect);
srcRect = Screen('Rect', tex);
imgW = srcRect(3) - srcRect(1);
imgH = srcRect(4) - srcRect(2);

scale = min([winW / imgW, winH / imgH, maxWidth / imgW, maxHeight / imgH]);
dstW = imgW * scale;
dstH = imgH * scale;
[xc, yc] = RectCenter(winRect);
destRect = CenterRectOnPointd([0 0 dstW dstH], xc, yc);
end

function blankRect = computeBlankRect(winRect, maxWidth, maxHeight)
% Compute the centered white rectangle used for blank trials.
winW = RectWidth(winRect);
winH = RectHeight(winRect);
rectW = min([winW, maxWidth]);
rectH = min([winH, maxHeight]);
[xCenter, yCenter] = RectCenter(winRect);
blankRect = CenterRectOnPointd([0 0 rectW rectH], xCenter, yCenter);
end

function tf = checkAbort()
% Return true if ESC is currently pressed.
tf = false;
[down, ~, keyCode] = KbCheck;
if down && keyCode(KbName('ESCAPE'))
    tf = true;
end
end

function blankIndices = generateBlankIndicesWithMinInterval(nTrials, nBlanks, minInterval)
% Generate blank-trial indices under a minimum-spacing constraint.
blankIndices = [];
effectiveTrials = nTrials - 2 * minInterval;
if effectiveTrials <= 0
    warning('NVDTask:BlankSpacing', ...
        'nTrials=%d is too small for minInterval=%d. No blank trials were inserted.', nTrials, minInterval);
    return;
end

maxPossibleBlanks = floor((effectiveTrials + minInterval) / (minInterval + 1));
if nBlanks > maxPossibleBlanks
    warning('NVDTask:BlankSpacing', ...
        'Requested %d blanks, but only %d can satisfy the spacing constraint.', nBlanks, maxPossibleBlanks);
    nBlanks = maxPossibleBlanks;
end

maxAttempts = 1000;
for attempt = 1:maxAttempts
    blankIndices = [];
    availablePositions = (1 + minInterval):(nTrials - minInterval);

    for iBlank = 1:nBlanks
        if isempty(availablePositions)
            break;
        end
        selectedIdx = randi(numel(availablePositions));
        selectedPos = availablePositions(selectedIdx);
        blankIndices(end + 1) = selectedPos; %#ok<AGROW>
        invalidRange = max(1 + minInterval, selectedPos - minInterval) : ...
            min(nTrials - minInterval, selectedPos + minInterval);
        availablePositions = setdiff(availablePositions, invalidRange);
    end

    if numel(blankIndices) == nBlanks
        break;
    end
end

blankIndices = sort(blankIndices);
end

function [responseKey, responseTime] = collectResponse(stimulusOnset)
% Collect the earliest numeric response from keys 1--4.
responseKey = "n/a";
responseTime = NaN;
[pressed, firstPress] = KbQueueCheck;
if ~pressed
    return;
end

targetKeys = {'1!', '2@', '3#', '4$', '1', '2', '3', '4'};
earliestTime = inf;
for iKey = 1:numel(targetKeys)
    keyCode = KbName(targetKeys{iKey});
    if firstPress(keyCode) > 0 && firstPress(keyCode) < earliestTime
        earliestTime = firstPress(keyCode);
        responseKey = string(targetKeys{iKey});
    end
end

if isfinite(earliestTime)
    responseTime = earliestTime - stimulusOnset;
end
end

function logMessage(logFile, verbose, message)
% Write a timestamped message to the log file and optionally to the command window.
timestamp = char(datetime('now', 'Format', 'HH:mm:ss'));
if ~isempty(logFile) && logFile ~= -1
    fprintf(logFile, '[%s] %s\n', timestamp, message);
end
if verbose
    fprintf('[%s] %s\n', timestamp, message);
end
end

function cleanupTask(logFile, debug)
% Close the log file, restore cursor state, and close Psychtoolbox windows.
if exist('logFile', 'var') && ~isempty(logFile) && logFile ~= -1
    fprintf(logFile, '\nTask ended at %s\n', char(datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss')));
    fclose(logFile);
end
if ~debug
    ShowCursor;
end
sca;
end
