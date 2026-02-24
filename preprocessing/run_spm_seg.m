% 批次處理版本：接收輸入影像、輸出資料夾、和原始檔名基底
function run_spm_seg(registered_input_path, output_dir, original_basename)
    % ---------------------------------------------------------------------
    % Part 1: SPM 分割任務設定 (這部分保持不變)
    % ---------------------------------------------------------------------
    matlabbatch{1}.spm.spatial.preproc.channel.vols = {[registered_input_path ',1']};
    matlabbatch{1}.spm.spatial.preproc.channel.biasreg = 0.0001;
    matlabbatch{1}.spm.spatial.preproc.channel.biasfwhm = 60;
    matlabbatch{1}.spm.spatial.preproc.channel.write = [0 0];
    matlabbatch{1}.spm.spatial.preproc.tissue(1).tpm = {'C:\matlab\toolbox\spm\tpm\TPM.nii,1'};
    matlabbatch{1}.spm.spatial.preproc.tissue(1).ngaus = 1;
    matlabbatch{1}.spm.spatial.preproc.tissue(1).native = [1 0];
    matlabbatch{1}.spm.spatial.preproc.tissue(1).warped = [0 0];
    matlabbatch{1}.spm.spatial.preproc.tissue(2).tpm = {'C:\matlab\toolbox\spm\tpm\TPM.nii,2'};
    matlabbatch{1}.spm.spatial.preproc.tissue(2).ngaus = 1;
    matlabbatch{1}.spm.spatial.preproc.tissue(2).native = [1 0];
    matlabbatch{1}.spm.spatial.preproc.tissue(2).warped = [0 0];
    matlabbatch{1}.spm.spatial.preproc.tissue(3).tpm = {'C:\matlab\toolbox\spm\tpm\TPM.nii,3'};
    matlabbatch{1}.spm.spatial.preproc.tissue(3).ngaus = 2;
    matlabbatch{1}.spm.spatial.preproc.tissue(3).native = [1 0];
    matlabbatch{1}.spm.spatial.preproc.tissue(3).warped = [0 0];
    matlabbatch{1}.spm.spatial.preproc.tissue(4).tpm = {'C:\matlab\toolbox\spm\tpm\TPM.nii,4'};
    matlabbatch{1}.spm.spatial.preproc.tissue(4).ngaus = 3;
    matlabbatch{1}.spm.spatial.preproc.tissue(4).native = [0 0];
    matlabbatch{1}.spm.spatial.preproc.tissue(4).warped = [0 0];
    matlabbatch{1}.spm.spatial.preproc.tissue(5).tpm = {'C:\matlab\toolbox\spm\tpm\TPM.nii,5'};
    matlabbatch{1}.spm.spatial.preproc.tissue(5).ngaus = 4;
    matlabbatch{1}.spm.spatial.preproc.tissue(5).native = [0 0];
    matlabbatch{1}.spm.spatial.preproc.tissue(5).warped = [0 0];
    matlabbatch{1}.spm.spatial.preproc.tissue(6).tpm = {'C:\matlab\toolbox\spm\tpm\TPM.nii,6'};
    matlabbatch{1}.spm.spatial.preproc.tissue(6).ngaus = 2;
    matlabbatch{1}.spm.spatial.preproc.tissue(6).native = [0 0];
    matlabbatch{1}.spm.spatial.preproc.tissue(6).warped = [0 0];
    matlabbatch{1}.spm.spatial.preproc.warp.mrf = 1;
    matlabbatch{1}.spm.spatial.preproc.warp.cleanup = 1;
    matlabbatch{1}.spm.spatial.preproc.warp.reg = [0 0 0.1 0.01 0.04];
    matlabbatch{1}.spm.spatial.preproc.warp.affreg = 'mni';
    matlabbatch{1}.spm.spatial.preproc.warp.fwhm = 0;
    matlabbatch{1}.spm.spatial.preproc.warp.samp = 3;
    matlabbatch{1}.spm.spatial.preproc.warp.write = [0 0];
    matlabbatch{1}.spm.spatial.preproc.warp.vox = NaN;
    matlabbatch{1}.spm.spatial.preproc.warp.bb = [NaN NaN NaN; NaN NaN NaN];
    
    % 設定SPM默認值並執行任務
    spm('defaults', 'FMRI');
    spm_jobman('run', matlabbatch);
    
    % ---------------------------------------------------------------------
    % Part 2: 移動並重新命名檔案 (使用新的邏輯)
    % ---------------------------------------------------------------------
    fprintf('SPM 處理完成。正在將結果移動並重新命名至: %s\n', output_dir);
    
    % SPM的輸出會存在和輸入影像(registered_input_path)相同的暫存資料夾中
    [temp_dir, temp_name, ~] = fileparts(registered_input_path);
    
    % 尋找所有由 SPM 產生的 c1, c2... 檔案
    search_pattern = fullfile(temp_dir, ['c*' temp_name '.nii']);
    files_to_move = dir(search_pattern);

    % 迴圈處理每一個找到的檔案
    for i = 1:length(files_to_move)
        source_file = fullfile(files_to_move(i).folder, files_to_move(i).name);
        
        % 根據原始檔名來建立新的檔名
        % 例如，如果找到的檔案是 'c1registered_for_matlab.nii'
        % 且 original_basename 是 'subject_001'
        % 新檔名將會是 'c1subject_001.nii'
        
        % 提取 'c1', 'c2' 等前綴
        prefix = regexp(files_to_move(i).name, '^c[1-6]', 'match', 'once');
        if ~isempty(prefix)
            new_filename = [prefix, original_basename, '.nii'];
            destination_file = fullfile(output_dir, new_filename);
            
            fprintf('正在移動 %s\n  至 %s\n', source_file, destination_file);
            movefile(source_file, destination_file);
        end
    end
    
    fprintf('所有檔案已成功移動！\n');
end