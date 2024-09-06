// Verilog code for FPGA to process EEG signals with filtering and signal normalization

module eeg_fpga_processor (
    input wire clk,                    // System clock
    input wire reset,                  // Reset signal
    input wire [7:0] eeg_data,         // 8-bit EEG data input
    output reg [7:0] processed_data,   // 8-bit processed EEG data output
    output wire data_ready             // Signal to indicate when processing is complete
);

    // Internal registers for filtering and signal processing
    reg [7:0] low_pass_data;
    reg [7:0] high_pass_data;
    reg [7:0] normalized_data;

    // Low-pass filter implementation
    always @(posedge clk or posedge reset) begin
        if (reset) begin
            low_pass_data <= 0;
        end else begin
            low_pass_data <= (eeg_data + low_pass_data) >> 1;  // Simple averaging filter for low-pass
        end
    end

    // High-pass filter implementation
    always @(posedge clk or posedge reset) begin
        if (reset) begin
            high_pass_data <= 0;
        end else begin
            high_pass_data <= eeg_data - low_pass_data;  // Subtract low-pass filtered data to get high-pass
        end
    end

    // Signal normalization (scale EEG signal to [0, 255] range)
    always @(posedge clk or posedge reset) begin
        if (reset) begin
            normalized_data <= 0;
        end else begin
            normalized_data <= (high_pass_data > 127) ? 127 : high_pass_data;  // Clamp values to avoid overflow
        end
    end

    // Output processed data
    always @(posedge clk or posedge reset) begin
        if (reset) begin
            processed_data <= 0;
        end else begin
            processed_data <= normalized_data;  // Output the normalized EEG data
        end
    end

    // Signal to indicate that data processing is complete
    assign data_ready = (processed_data != 0);

endmodule