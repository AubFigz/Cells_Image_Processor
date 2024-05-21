import subprocess
import os
import logging
import shutil
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.sparse.linalg import svds
import glob
import cooler

class OmicsDataset:
    """
    Base class for omics data handling. This class provides foundational methods and structures
    for loading and preprocessing omics datasets. It is designed to be extended by subclasses
    to handle specific types of omics data, such as genomic, transcriptomic, proteomic, etc.

    Attributes:
        file_path (str): Path to the primary data file associated with the dataset.
        raw_data (pandas.DataFrame, optional): Container for raw data loaded from the file_path.
            Initialized as None and should be loaded or assigned in subclass implementations.
        processed_data (pandas.DataFrame, optional): Container for data after preprocessing steps
            have been applied. Initialized as None and should be generated through subclass methods.
    """

    def __init__(self, file_path):
        """
        Initializes the OmicsDataset with a path to the dataset file.

        Args:
            file_path (str): Path to the file containing the dataset.
        """
        self.file_path = file_path
        self.raw_data = None
        self.processed_data = None

    @staticmethod
    def load_data_from_csv(file_path, separator=',', error_message="Error loading CSV file"):
        """
        Utility function to load data from a CSV file. This method is static, meaning it can be
        called without an instance of OmicsDataset. It is intended to be used by subclasses or
        directly for loading CSV data into a pandas DataFrame.

        Args:
            file_path (str): Path to the CSV file to be loaded.
            error_message (str): Error message to display if loading fails. Defaults to a generic message.

        Returns:
            pandas.DataFrame or None: The loaded data as a pandas DataFrame, or None if an error occurs.
        """
        try:
            return pd.read_csv(file_path, sep=separator)
        except FileNotFoundError:
            logging.error(f"{error_message}: {file_path}")
            return None

    def preprocess_data(self):
        """
        Template method for data preprocessing. This method is intended to be overridden by subclasses
        to implement specific preprocessing steps relevant to the type of omics data being handled.

        The method should update `self.processed_data` with the preprocessed data.
        """
        pass

    def load_data(self):
        """
        Template method for loading data. This method is intended to be overridden by subclasses
        to implement specific data loading logic, depending on the data format and requirements.

        The method should update `self.raw_data` with the loaded data.
        """
        pass

class TranscriptomicsData:
    # Setup class-level logger
    logger = logging.getLogger('TranscriptomicsDataLogger')
    logger.setLevel(logging.DEBUG)  # Capture all levels of logs

    # Define a file handler for debug logs
    debug_handler = logging.FileHandler('transcriptomics_debug.log', mode='w')
    debug_handler.setLevel(logging.DEBUG)
    debug_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    debug_handler.setFormatter(debug_formatter)
    logger.addHandler(debug_handler)

    # Define a file handler for error logs
    error_handler = logging.FileHandler('transcriptomics_error.log', mode='w')
    error_handler.setLevel(logging.ERROR)
    error_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    error_handler.setFormatter(error_formatter)
    logger.addHandler(error_handler)

    def __init__(self, file_path, gene_length_file_path, qc_dir="qc_reports"):
        """
        Initializes the class with necessary paths and sets up the environment for processing.
        Args:
            file_path (str): Path to the raw FASTQ file(s).
            gene_length_file_path (str): Path to the file with gene length information.
            qc_dir (str): Directory for quality control outputs.
        """
        self.file_path = file_path
        self.gene_length_file_path = gene_length_file_path
        self.qc_dir = qc_dir
        self.gene_lengths_df = self._fetch_gene_lengths()

        # Ensure the QC directory exists
        if not os.path.exists(self.qc_dir):
            os.makedirs(self.qc_dir)
        self.logger.debug(f"Initializing TranscriptomicsData with file_path: {file_path}")

    def _ensure_tool_availability(self, tool_name):
        """
        Checks if a tool is available in the PATH.
        Args:
            tool_name (str): The tool to check.
        Raises:
            EnvironmentError: If the tool is not found.
        """
        if not shutil.which(tool_name):
            msg = f"{tool_name} is required but not available in PATH."
            self.logger.error(msg)
            raise EnvironmentError(msg)

    def run_fastqc(self, additional_options=None):
        """
        Performs quality control on the raw FASTQ files using FastQC.
        Args:
            additional_options (list): Additional options for FastQC.
        """
        self._ensure_tool_availability('fastqc')
        if not os.path.exists(self.qc_dir):
            os.makedirs(self.qc_dir)
        cmd = ['fastqc', self.file_path, '-o', self.qc_dir] + (additional_options or [])
        try:
            subprocess.run(cmd, check=True)
            self.logger.info("FastQC completed successfully.")
        except subprocess.CalledProcessError as e:
            self.logger.error("FastQC failed.", exc_info=True)

    def trim_reads(self):
        """
        Trims adapters and low-quality sequences from the FASTQ files using Trim Galore.
        """
        self._ensure_tool_availability('trim_galore')
        cmd = ['trim_galore', '--output_dir', self.qc_dir, self.file_path]
        try:
            subprocess.run(cmd, check=True)
            self.logger.info("Read trimming completed successfully.")
        except subprocess.CalledProcessError:
            self.logger.error("Read trimming failed.", exc_info=True)

    def _fetch_gene_lengths(self):
        """
        Fetches gene length information from a CSV file.
        Returns:
            A DataFrame with gene lengths.
        """
        try:
            return pd.read_csv(self.gene_length_file_path)
        except FileNotFoundError:
            self.logger.error(f"Gene length file not found: {self.gene_length_file_path}", exc_info=True)
            return pd.DataFrame()

    def normalize_data(self):
        """
        Normalizes raw count data to TPM (Transcripts Per Million).
        """
        if self.gene_lengths_df.empty:
            self.logger.error("Gene length data is required for normalization.")
            return

        # Assuming `self.raw_data` is a DataFrame loaded with raw counts data
        # and `self.gene_lengths_df` is formatted as 'gene_id', 'length'
        merged_data = pd.merge(self.raw_data, self.gene_lengths_df, on='gene_id')
        rpk = merged_data['counts'] / (merged_data['length'] / 1000)  # Reads Per Kilobase
        tpm = rpk / rpk.sum() * 1e6  # Transcripts Per Million
        merged_data['TPM'] = tpm
        self.processed_data = merged_data[['gene_id', 'TPM']]
        self.logger.info("Normalization to TPM completed.")

    def quantify_expression(self, fastq_files, output_dir, quantifier='salmon', additional_options=None):
        self._ensure_tool_availability(quantifier)
        if quantifier == 'salmon':
            cmd = ['salmon', 'quant', '-i', self.reference_index, '-l', 'A',
                   '-1', fastq_files[0], '-2', fastq_files[1], '-o', output_dir] + (additional_options or [])
        try:
            subprocess.run(cmd, check=True)
            self.logger.info(f"{quantifier} quantification completed successfully.")
        except subprocess.CalledProcessError:
            self.logger.error(f"{quantifier} quantification failed.", exc_info=True)

    def align_reads(self, fastq_files, output_bam, aligner='hisat2', additional_options=None):
        self._ensure_tool_availability(aligner)
        if aligner == 'hisat2':
            cmd = ['hisat2', '-x', self.reference_genome, '-1', fastq_files[0], '-2', fastq_files[1], '|',
                   'samtools', 'view', '-bS', '-', '>', output_bam] + (additional_options or [])
        try:
            subprocess.run(" ".join(cmd), shell=True, check=True)  # Use shell=True cautiously
            self.logger.info(f"Alignment with {aligner} completed successfully.")
        except subprocess.CalledProcessError:
            self.logger.error(f"Alignment with {aligner} failed.", exc_info=True)

    def run_advanced_qc(self, bam_file, qc_tool='qualimap', output_dir=None):
        """
        Runs advanced quality control analysis on BAM files.

        Args:
            bam_file (str): Path to the BAM file to analyze.
            qc_tool (str): The quality control tool to use. Defaults to 'qualimap'.
            output_dir (str, optional): The directory where the QC reports will be saved. Defaults to the class's qc_dir attribute.
        """
        self._ensure_tool_availability(qc_tool)
        if qc_tool == 'qualimap':
            cmd = ['qualimap', 'rnaseq', '-bam', bam_file, '-outdir', output_dir or self.qc_dir]
        try:
            subprocess.run(cmd, check=True)
            self.logger.info(f"{qc_tool} QC analysis completed successfully.")
        except subprocess.CalledProcessError as e:
            self.logger.error(f"{qc_tool} QC analysis failed.", exc_info=True)

    def filter_low_expressed_genes(self, expression_matrix, threshold=1):
        """
        Filters out genes with expression levels below a specified threshold.

        Args:
            expression_matrix (pd.DataFrame): The expression matrix to filter.
            threshold (float, optional): The expression threshold below which genes are filtered out. Defaults to 1.

        Returns:
            pd.DataFrame: The filtered expression matrix.
        """
        filtered_matrix = expression_matrix[expression_matrix.sum(axis=1) > threshold]
        self.logger.info("Filtered lowly expressed genes.")
        return filtered_matrix

    def generate_multiqc_report(self, analysis_dir):
        """
        Generates a consolidated report using MultiQC for all analyses.

        Args:
            analysis_dir (str): The directory containing analysis results to be compiled by MultiQC.
        """
        self._ensure_tool_availability('multiqc')
        cmd = ['multiqc', analysis_dir, '-o', self.qc_dir]
        try:
            subprocess.run(cmd, check=True)
            self.logger.info("MultiQC report generated successfully.")
        except subprocess.CalledProcessError as e:
            self.logger.error("MultiQC report generation failed.", exc_info=True)

    def preprocess_data(self, additional_fastqc_options=None, normalization_method='TPM',
                        alignment_tool='hisat2', quantification_tool='salmon',
                        perform_advanced_qc=True, quantifier_options=None):
        """
        Orchestrates the preprocessing of transcriptomics data, including quality control,
        read trimming, expression quantification, normalization, and optional advanced QC.

        Args:
            additional_fastqc_options (list, optional): Additional command-line options for FastQC.
            normalization_method (str, optional): Method for normalizing gene expression data. Defaults to 'TPM'.
            alignment_tool (str, optional): Alignment tool to use. Set to None if alignment is not required.
            quantification_tool (str, optional): Tool for quantifying expression levels.
            perform_advanced_qc (bool, optional): Whether to perform advanced quality control. Defaults to True.
            quantifier_options (list, optional): Additional command-line options for the quantification tool.
        """
        # Initial quality control with FastQC
        self.run_fastqc(additional_options=additional_fastqc_options)

        # Trimming adapters and low-quality sequences from the FASTQ files
        trimmed_files = self.trim_reads()  # Implement handling for paired-end reads within trim_reads
        if not trimmed_files:
            self.logger.error("Trimming failed. Preprocessing aborted.")
            return

        # Post-trim quality control to assess the quality of trimmed sequences
        self.post_trim_fastqc(trimmed_files, additional_options=additional_fastqc_options)

        # Alignment is conducted only if the selected quantification tool requires aligned reads
        if alignment_tool and quantification_tool in ['featureCounts', 'HTSeq']:
            self.align_reads(trimmed_files, alignment_tool=alignment_tool,
                             additional_options=quantifier_options)

        # Quantification of expression levels
        self.quantify_expression(trimmed_files, quantification_tool=quantification_tool,
                                 additional_options=quantifier_options)

        # Loading quantified data for normalization
        # Ensure the quantification output is compatible with pd.read_csv
        self.load_quantified_data("path/to/quantification_results.csv")

        # Normalize the expression data
        self.normalize_data(normalization_method=normalization_method)

        # Performing advanced QC if requested
        if perform_advanced_qc:
            self.run_advanced_qc("path/to/aligned_reads.bam", qc_tool='qualimap')

        # Filtering out lowly expressed genes
        self.filter_low_expressed_genes(self.processed_data)

        # Generate a consolidated report with MultiQC
        self.generate_multiqc_report("path/to/analysis_results")

        self.logger.info("Preprocessing and initial analyses completed successfully.")

class ChIPSeqData(OmicsDataset):
    def __init__(self, file_path, reference_genome_path, qc_dir="qc_reports"):
    super().__init__(file_path)
    self.reference_genome_path = reference_genome_path
    self.qc_dir = qc_dir
    # Ensure MACS2 is available; similar checks could be added for other tools like BWA if used for alignment
    self._ensure_tool_availability("macs2")

    def _ensure_tool_availability(self, tool_name):
        if not shutil.which(tool_name):
            raise EnvironmentError(f"{tool_name} is required but not available in PATH.")

    def run_fastqc(self, input_dir, output_dir=None):
        """
        Run FastQC for initial read quality assessment on files within the specified directory.

        Args:
            input_dir (str): Directory containing FASTQ files for QC.
            output_dir (str, optional): Directory to store FastQC reports. Defaults to self.qc_dir.
        """
        output_dir = output_dir or self.qc_dir
        fastqc_cmd = ['fastqc', '-o', output_dir, input_dir + '/*.fastq']
        try:
            subprocess.run(fastqc_cmd, check=True)
            logging.info("FastQC completed successfully.")
        except subprocess.CalledProcessError as e:
            logging.error(f"FastQC failed: {e}")

    def align_reads(self):
        """
        Align reads to the reference genome using the specified aligner, and update self.bam_file.
        """
        # Ensure BWA is available
        self._ensure_tool_availability("bwa")

        # Assuming self.file_path is the path to the FASTQ file
        aligned_sam = self.file_path.replace('.fastq', '.aligned.sam')
        sorted_bam = self.file_path.replace('.fastq', '.sorted.bam')
        self.bam_file = sorted_bam  # Update class attribute

        # Perform alignment
        bwa_cmd = ['bwa', 'mem', self.reference_genome_path, self.file_path, '-o', aligned_sam]
        subprocess.run(bwa_cmd, check=True)

        # Sort and convert SAM to BAM
        sort_cmd = ['samtools', 'sort', '-o', sorted_bam, aligned_sam]
        subprocess.run(sort_cmd, check=True)

        # Remove the SAM file to save space
        os.remove(aligned_sam)
        logging.info("Alignment and sorting completed successfully.")

    def run_qualimap(self, input_bam, output_dir=None):
        """
        Assess the quality of alignment using QualiMap on the specified BAM file.

        Args:
            input_bam (str): Path to the BAM file to analyze.
            output_dir (str, optional): Directory to store QualiMap reports. Defaults to self.qc_dir.
        """
        output_dir = output_dir or self.qc_dir
        qualimap_cmd = ['qualimap', 'bamqc', '-bam', input_bam, '-outdir', output_dir]
        try:
            subprocess.run(qualimap_cmd, check=True)
            logging.info("QualiMap analysis completed successfully.")
        except subprocess.CalledProcessError as e:
            logging.error(f"QualiMap analysis failed: {e}")

    def _call_peaks_with_macs2(self, input_bam, output_dir, params=None):
        """
        Call peaks using MACS2 with the provided parameters on the specified BAM file.

        Args:
            input_bam (str): Path to the BAM file for peak calling.
            output_dir (str): Directory to store MACS2 output.
            params (dict, optional): Additional command-line parameters for MACS2.
        """
        self._ensure_tool_availability("macs2")

        macs2_command = ['macs2', 'callpeak', '-t', input_bam, '--outdir', output_dir]
        if params:
            for param, value in params.items():
                macs2_command.extend([param, str(value)])

        subprocess.run(macs2_command, check=True)
        logging.info("MACS2 peak calling completed successfully.")

class CUTRUNData(OmicsDataset):
    def __init__(self, file_path, control_path='non', qc_dir="qc_reports"):
        """
        Initialize the CUTRUNData class with file paths for CUT&RUN and control data, along with a QC directory.

        Args:
            file_path (str): Path to the CUT&RUN sequencing data file.
            control_path (str): Path to the control (input) data file, if available. Defaults to 'non' for no control.
            qc_dir (str): Directory where QC reports will be stored. Defaults to 'qc_reports'.
        """
        super().__init__(file_path)
        self.control_path = control_path
        self.qc_dir = qc_dir
        # Directly ensure SEACR tool's availability during object initialization
        self._ensure_tool_availability("SEACR")

    def _ensure_tool_availability(self, tool_name):
        """
        Ensure an external tool is available in the system's PATH and raise an exception if it is not.

        Args:
            tool_name (str): The name of the tool to check.

        Raises:
            EnvironmentError: If the specified tool is not found in the system's PATH.
        """
        if not shutil.which(tool_name):
            raise EnvironmentError(f"{tool_name} is required but not available in PATH.")

    def run_fastqc(self):
        """Perform initial quality control on raw sequencing data using FastQC."""
        if not os.path.exists(self.qc_dir):
            os.makedirs(self.qc_dir)
        fastqc_cmd = ['fastqc', self.file_path, '-o', self.qc_dir]
        try:
            subprocess.run(fastqc_cmd, check=True)
            logging.info("FastQC completed successfully.")
        except subprocess.CalledProcessError as e:
            logging.error(f"FastQC failed: {e}")

    def run_qualimap(self):
        """Evaluate alignment quality using QualiMap on BAM files produced after aligning CUT&RUN data."""
        qualimap_cmd = ['qualimap', 'bamqc', '-bam', self.bam_file, '-outdir', self.qc_dir]
        try:
            subprocess.run(qualimap_cmd, check=True)
            logging.info("QualiMap analysis completed successfully.")
        except subprocess.CalledProcessError as e:
            logging.error(f"QualiMap analysis failed: {e}")

    def _call_peaks_with_seacr(self, threshold='0.01', norm_method='norm', stringent=False):
        self._ensure_tool_availability("SEACR")

        output_file = f"{os.path.splitext(self.file_path)[0]}_SEACR_peaks.bed"
        stringent_flag = 'stringent' if stringent else 'relaxed'
        seacr_command = [
            'SEACR', self.file_path, self.control_path,
            threshold, norm_method, stringent_flag, output_file
        ]
        subprocess.run(seacr_command, check=True)
        logging.info("SEACR peak calling completed successfully.")

        # Return or process the peak data
        peaks_df = pd.read_csv(output_file, sep='\t', header=None)
        return peaks_df

class HiCData(OmicsDataset):
    def __init__(self, file_path):
        """
        Initialize the HiCData class with the path to a .cool file containing Hi-C data.

        Args:
            file_path (str): Path to the .cool file.
        """
        super().__init__(file_path)
        self.processed_data = None  # Initialize processed_data

    def _load_hic_data(self):
        """
        Load Hi-C data from a .cool file and return as a cooler.Cooler object.

        Returns:
            cooler.Cooler: Cooler object containing the Hi-C matrix if successful, None otherwise.
        """
        try:
            hic_matrix = cooler.Cooler(self.file_path)
            logging.info("Hi-C data loaded successfully.")
            return hic_matrix
        except Exception as e:
            logging.error(f"Loading Hi-C data failed: {e}")
            return None

    def _normalize_matrix_ice(self, hic_matrix):
        """
        Normalize the Hi-C matrix using the Iterative Correction and Eigenvector decomposition (ICE) method.

        Args:
            hic_matrix (cooler.Cooler): Cooler object to normalize.

        Returns:
            scipy.sparse.csr_matrix: Normalized sparse matrix if successful, None otherwise.
        """
        try:
            # Ensure that 'balance' is calculated if not done before
            if not hic_matrix.bins()['weight'].notnull().all():
                hic_matrix = cooler.balance_cooler(hic_matrix)
            matrix = hic_matrix.matrix(balance=True, sparse=True)[:]
            logging.info("ICE normalization completed. Matrix remains sparse.")
            return matrix
        except Exception as e:
            logging.error(f"ICE normalization failed: {e}")
            return None

    def _analyze_compartments(self, hic_matrix):
        """
        Perform compartment analysis on the normalized Hi-C matrix using PCA.

        Args:
            hic_matrix (scipy.sparse.csr_matrix): Normalized Hi-C matrix.

        Returns:
            np.ndarray: The first principal component, indicative of A/B compartments.
        """
        # Convert the sparse matrix to a dense format for PCA
        dense_matrix = np.nan_to_num(hic_matrix.todense())

        # Standardize the matrix
        scaler = StandardScaler(with_mean=False)
        scaled_matrix = scaler.fit_transform(dense_matrix)

        # Perform Singular Value Decomposition (SVD) to get the first principal component
        u, s, vt = svds(scaled_matrix, k=1)
        first_principal_component = u[:, 0]

        # The sign of the principal component can be arbitrary, so we align it with the mean
        if np.mean(first_principal_component) < 0:
            first_principal_component = -first_principal_component

        logging.info("Compartment analysis completed using PCA.")

        return first_principal_component

    def preprocess_data(self, normalization_method='ICE', perform_compartment_analysis=False):
        """
        Preprocess Hi-C data, including matrix normalization and optional compartment analysis.

        Args:
            normalization_method (str): Normalization method to use, default is 'ICE'.
            perform_compartment_analysis (bool): Whether to perform compartment analysis, default is False.
        """
        hic_matrix = self._load_hic_data()
        if hic_matrix is None:
            logging.error("Failed to load Hi-C data.")
            return

        if normalization_method == 'ICE':
            hic_matrix = self._normalize_matrix_ice(hic_matrix)
            if hic_matrix is None:
                logging.error("Normalization failed.")
                return

        if perform_compartment_analysis:
            compartments = self._analyze_compartments(hic_matrix)
            # You might want to attach the compartments to bins or save them for further analysis
            self.processed_data = {'hic_matrix': hic_matrix, 'compartments': compartments}
        else:
            self.processed_data = hic_matrix

class NanoporeData(OmicsDataset):
    def __init__(self, fastq_dir, reference_genome, qc_dir="qc_reports"):
        """
        Initializes the NanoporeData class for processing Nanopore sequencing data.

        Args:
            fastq_dir (str): Directory containing raw or basecalled Nanopore FASTQ files.
            reference_genome (str): Path to the reference genome file for read alignment.
            qc_dir (str): Directory for storing quality control outputs. Defaults to 'qc_reports'.
        """
        super().__init__(fastq_dir)
        self.reference_genome = reference_genome
        self.qc_dir = qc_dir  # QC directory initialization
        self.bam_file = os.path.join(qc_dir, "aligned_reads.bam")  # Defines BAM file path
        self.variants_vcf = os.path.join(qc_dir, "variants.vcf")  # Defines VCF file path for detected variants

        # Check for the availability of required external tools
        for tool in ["guppy_basecaller", "minimap2", "sniffles", "fastqc"]:
            if not self.check_external_tool_availability(tool):
                raise EnvironmentError(f"{tool} is required but not available.")

    def check_external_tool_availability(self, tool_name):
        """
        Checks if an external tool is available in the system's PATH.

        Args:
            tool_name (str): The name of the tool to check.

        Returns:
            bool: True if the tool is available, False otherwise.
        """
        return shutil.which(tool_name) is not None

    def run_pre_basecalling_qc(self, raw_data_dir):
        """
        Performs quality control on raw Nanopore data files using NanoPlot or PycoQC.

        Args:
            raw_data_dir (str): Directory containing raw Nanopore data files.
        """
        # Example command for NanoPlot; adjust according to actual usage and available tools
        nanoplot_cmd = ["NanoPlot", "--fastq_rich", raw_data_dir, "-o", self.qc_dir]
        try:
            subprocess.run(nanoplot_cmd, check=True)
            logging.info("NanoPlot QC analysis completed successfully.")
        except subprocess.CalledProcessError as e:
            logging.error(f"NanoPlot QC analysis failed: {e}")

    def basecall_with_guppy(self, raw_data_dir, output_dir, config, flowcell, kit):
        """
        Performs basecalling on raw Nanopore data using Guppy.

        Args:
            raw_data_dir (str): Directory containing raw Nanopore data files.
            output_dir (str): Directory to save basecalled FASTQ files.
            config (str): Guppy config suitable for the sequencing kit used.
            flowcell (str): Flowcell type.
            kit (str): Sequencing kit type.
        """
        guppy_cmd = [
            "guppy_basecaller",
            "-i", raw_data_dir,
            "-s", output_dir,
            "--config", config,
            "--flowcell", flowcell,
            "--kit", kit
        ]
        try:
            subprocess.run(guppy_cmd, check=True, capture_output=True)
            logging.info("Guppy basecalling completed successfully.")
        except subprocess.CalledProcessError as e:
            logging.error(f"Guppy basecalling failed: {e.stderr.decode()}")

    def run_post_basecalling_qc(self, fastq_dir):
        """
        Performs quality control on basecalled FASTQ files using FastQC.

        Args:
            fastq_dir (str): Directory containing basecalled FASTQ files.
        """
        fastqc_cmd = ['fastqc', fastq_dir + '/*.fastq', '-o', self.qc_dir]
        try:
            subprocess.run(fastqc_cmd, check=True)
            logging.info("Post-basecalling FastQC analysis completed successfully.")
        except subprocess.CalledProcessError as e:
            logging.error(f"Post-basecalling FastQC analysis failed: {e}")

    def _align_reads(self):
        """
        Aligns base-called FASTQ reads to the reference genome using Minimap2.
        """
        # Assuming fastq_dir contains the basecalled FASTQ files
        fastq_files = glob.glob(f"{self.fastq_dir}/*.fastq")
        if not fastq_files:
            logging.error("No FASTQ files found in the specified directory.")
            return False

        align_cmd = ["minimap2", "-ax", "map-ont",

    def _detect_structural_variants(self):
        """
        Detects structural variants from aligned reads using Sniffles.
        """
        # Ensure the BAM file exists and is not empty
        if not os.path.exists(self.bam_file) or os.path.getsize(self.bam_file) == 0:
            logging.error("Aligned BAM file does not exist or is empty.")
            return False

        sv_call_cmd = ["sniffles", "-m", self.bam_file, "-v", self.variants_vcf]
        try:
            subprocess.run(sv_call_cmd, check=True)
            logging.info("Structural variant detection with Sniffles completed successfully.")
        except subprocess.CalledProcessError as e:
            logging.error(f"Structural variant detection failed with Sniffles: {e.stderr.decode()}")
            return False
        return True

    def preprocess_data(self):
        """
        Orchestrates the preprocessing of Nanopore sequencing data.
        This includes basecalling, read alignment, structural variant detection, and relevant QC steps.
        """
        # Define your paths and parameters for Guppy basecalling
        raw_data_dir = "path/to/raw_data"
        output_dir = "path/to/output_fastq"
        config = "dna_r9.4.1_450bps_hac.cfg"
        flowcell = "FLO-MIN106"
        kit = "SQK-LSK109"

        # Perform pre-basecalling QC
        self.run_pre_basecalling_qc(raw_data_dir)

        # Basecalling with Guppy
        self.basecall_with_guppy(raw_data_dir, output_dir, config, flowcell, kit)

        # Perform post-basecalling QC
        self.run_post_basecalling_qc(output_dir)

        # Align reads
        if not self._align_reads():
            logging.error("Read alignment process failed.")
            return

        # Detect structural variants
        if not self._detect_structural_variants():
            logging.error("Structural variant detection process failed.")

class WGSData(OmicsDataset):
    def __init__(self, fastq_files, reference_genome, qc_dir="qc_reports"):
        """
        Initialize the WGSData class for Whole Genome Sequencing data processing.
        Includes steps from quality control to variant calling.
        """
        super().__init__(fastq_files)
        self.reference_genome = reference_genome
        self.qc_dir = qc_dir
        if not os.path.exists(self.qc_dir):
            os.makedirs(self.qc_dir)
        self.bam_file = os.path.join(self.qc_dir, "aligned_reads.bam")
        self.variants_vcf = os.path.join(self.qc_dir, "variants.vcf")

    def _run_command(self, command, stdin=None):
        """
        Execute a system command with subprocess and handle errors.
        """
        try:
            subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=stdin)
            logging.info(f"Command executed successfully: {' '.join(command)}")
        except subprocess.CalledProcessError as e:
            logging.error(f"Command execution failed: {' '.join(command)}. Error: {e.stderr.decode()}")
            raise

    def run_fastqc(self):
        """
        Execute FastQC on input FASTQ files for initial quality assessment.
        """
        logging.info("Starting FastQC quality control...")
        for fastq_file in self.fastq_files:
            fastqc_cmd = ["fastqc", fastq_file, "-o", self.qc_dir]
            self._run_command(fastqc_cmd)
        logging.info("FastQC analysis completed for all files.")

    def trim_reads(self, adapter_sequence=None):
        """
        Trim adapters and low-quality sequences using Trim Galore!.
        """
        logging.info("Starting read trimming with Trim Galore!...")
        for fastq_file in self.fastq_files:
            trim_galore_cmd = ["trim_galore", "--output_dir", self.qc_dir, fastq_file]
            if adapter_sequence:
                trim_galore_cmd.extend(["--adapter", adapter_sequence])
            self._run_command(trim_galore_cmd)
        logging.info("Trimming completed.")

    def align_reads(self):
        """
        Align trimmed FASTQ files to the reference genome using BWA, followed by sorting with samtools.
        """
        logging.info("Starting read alignment with BWA and samtools...")
        for fastq_file in self.fastq_files:
            # Assuming single-end reads for simplicity; adjust for paired-end if needed
            align_cmd = ["bwa", "mem", "-t", "4", self.reference_genome, fastq_file]
            aligned_sam = fastq_file.replace('.fastq', '.sam')
            with open(aligned_sam, 'w') as sam_output:
                subprocess.run(align_cmd, stdout=sam_output)
            # Sort the SAM file and convert to BAM
            sort_cmd = ["samtools", "sort", "-o", self.bam_file, aligned_sam]
            self._run_command(sort_cmd)
            os.remove(aligned_sam)  # Clean up SAM file
        logging.info("Alignment and sorting completed.")

    def run_qualimap(self):
        """
        Execute QualiMap on the aligned BAM file to assess alignment quality.
        """
        logging.info("Starting QualiMap for post-alignment quality control...")
        qualimap_cmd = ["qualimap", "bamqc", "-bam", self.bam_file, "-outdir", self.qc_dir]
        self._run_command(qualimap_cmd)
        logging.info("QualiMap analysis completed.")

    def call_variants(self):
        """
        Call variants using the aligned reads with bcftools.
        """
        logging.info("Starting variant calling with bcftools...")
        mpileup_cmd = ["bcftools", "mpileup", "-Ou", "-f", self.reference_genome, self.bam_file]
        call_cmd = ["bcftools", "call", "--ploidy", "1", "-mv", "-o", self.variants_vcf]
        # Pipe mpileup output to call command
        mpileup_proc = subprocess.Popen(mpileup_cmd, stdout=subprocess.PIPE)
        self._run_command(call_cmd, stdin=mpileup_proc.stdout)
        mpileup_proc.stdout.close()
        logging.info("Variant calling completed.")

    def preprocess_data(self):
        """
        Orchestrates the WGS data processing pipeline, including QC steps, read trimming, read alignment,
        post-alignment QC, and variant calling.
        """
        logging.info("Beginning WGS data preprocessing pipeline...")
        self.run_fastqc()
        self.trim_reads()  # Optionally pass adapter_sequence if specific trimming is required
        self.align_reads()
        self.run_qualimap()
        self.call_variants()
        logging.info("WGS data processing pipeline with QC steps completed successfully.")
