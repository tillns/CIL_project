
#include <math.h>


void compute_hist(float * mag_arr, float * angle_arr, float * hist, int num_pixels, int num_bins){


	float mag, angle;
	
	float frac_index;
	int idx_l, idx_h;
	
	float w_l, w_h;
	
	
	float num_bins_f = (float)num_bins;
	float angle_per_bin = 180.0 / num_bins_f;


	for(int i = 0; i < num_pixels; i++) {
		
		
		mag = mag_arr[i];
		angle = angle_arr[i];
		
		
		if(angle > 180.0) {
			angle = angle - 180.0;
		}
		
		
		// this should be inside [0.0, num_bins_f]
		frac_index = angle / angle_per_bin; 
		
		
		if(frac_index <= (num_bins_f - 1.0)) {
			
			idx_l = (int)floor(frac_index);
			idx_h = (int)ceil(frac_index);
			
			if(idx_l != idx_h) {
				
				w_l = frac_index - floor(frac_index);
				w_h = ceil(frac_index) - frac_index;
				
				hist[idx_l] += 1.0 * w_l; // mag * w_l;
				hist[idx_h] += 1.0 * w_h; // mag * w_h;
				
			} else {
				
				hist[idx_l] += 1.0; // mag;
				
			}
			
		} else { // (num_bins_f - 1.0) < frac_index <= num_bins_f
			
			if(frac_index == num_bins_f) {
				
				hist[0] += 1.0; // mag;
				
			} else {
				
				w_l = frac_index - (num_bins_f - 1.0);
				w_h = num_bins_f - frac_index;
				
				hist[num_bins - 1] += 1.0 * w_l; // mag * w_l;
				hist[0] += 1.0 * w_h; // mag * w_h;
				
			}		
		}
	
	}

	
}
