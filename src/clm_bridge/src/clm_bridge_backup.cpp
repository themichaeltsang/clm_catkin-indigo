///////////////////////////////////////////////////////////////////////////////
// Copyright (C) 2015, University of Cambridge,
// all rights reserved.
//
// THIS SOFTWARE IS PROVIDED “AS IS” AND ANY EXPRESS OR IMPLIED WARRANTIES,
// INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY
// AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL
// THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
// INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY. OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
// ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Notwithstanding the license granted herein, Licensee acknowledges that certain components
// of the Software may be covered by so-called “open source” software licenses (“Open Source
// Components”), which means any software licenses approved as open source licenses by the
// Open Source Initiative or any substantially similar licenses, including without limitation any
// license that, as a condition of distribution of the software licensed under such license,
// requires that the distributor make the software available in source code format. Licensor shall
// provide a list of Open Source Components for a particular version of the Software upon
// Licensee’s request. Licensee will comply with the applicable terms of such licenses and to
// the extent required by the licenses covering Open Source Components, the terms of such
// licenses will apply in lieu of the terms of this Agreement. To the extent the terms of the
// licenses applicable to Open Source Components prohibit any of the restrictions in this
// License Agreement with respect to such Open Source Component, such restrictions will not
// apply to such Open Source Component. To the extent the terms of the licenses applicable to
// Open Source Components require Licensor to make an offer to provide source code or
// related information in connection with the Software, such offer is hereby made. Any request
// for source code or related information should be directed to cl-face-tracker-distribution@lists.cam.ac.uk
// Licensee acknowledges receipt of notices for the Open Source Components for the initial
// delivery of the Software.

//     * Any publications arising from the use of this software, including but
//       not limited to academic journal and conference publications, technical
//       reports and manuals, must cite one of the following works (the related one preferrably):
//
//       Tadas Baltrusaitis, Peter Robinson, and Louis-Philippe Morency. 3D
//       Constrained Local Model for Rigid and Non-Rigid Facial Tracking.
//       IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2012.    
//
//       Tadas Baltrusaitis, Peter Robinson, and Louis-Philippe Morency. 
//       Constrained Local Neural Fields for robust facial landmark detection in the wild.
//       in IEEE Int. Conference on Computer Vision Workshops, 300 Faces in-the-Wild Challenge, 2013.    
//
//       Tadas Baltrusaitis, Marwa Mahmoud, and Peter Robinson.
//		 Cross-dataset learning and person-specific normalisation for automatic Action Unit detection
//       Facial Expression Recognition and Analysis Challenge 2015,
//       IEEE International Conference on Automatic Face and Gesture Recognition, 2015
//
//       Erroll Wood, Tadas Baltrušaitis, Xucong Zhang, Yusuke Sugano, Peter Robinson, and Andreas Bulling
//		 Rendering of Eyes for Eye-Shape Registration and Gaze Estimation
//       in IEEE International. Conference on Computer Vision (ICCV), 2015
//
///////////////////////////////////////////////////////////////////////////////


// FeatureExtraction.cpp : Defines the entry point for the feature extraction console application.
#include "CLM_core.h"

#include <fstream>
#include <sstream>

#include <opencv2/videoio/videoio.hpp>  // Video write
#include <opencv2/videoio/videoio_c.h>  // Video write

#include <Face_utils.h>
#include <FaceAnalyser.h>
#include <GazeEstimation.h>

#include <ros/ros.h>
#include <clm_bridge/ClmHeads.h>
//#include <clm_bridge/ClmEyeGaze.h>
//#include <clm_bridge/ClmFacialActionUnit.h>



#include <filesystem.hpp>
#include <filesystem/fstream.hpp>

#define INFO_STREAM( stream ) \
std::cout << stream << std::endl

#define WARN_STREAM( stream ) \
std::cout << "Warning: " << stream << std::endl

#define ERROR_STREAM( stream ) \
std::cout << "Error: " << stream << std::endl

static void printErrorAndAbort( const std::string & error )
{
    std::cout << error << std::endl;
    abort();
}

#define FATAL_STREAM( stream ) \
printErrorAndAbort( std::string( "Fatal error: " ) + stream )

using namespace std;
using namespace cv;

using namespace boost::filesystem;

vector<string> get_arguments(int argc, char **argv)
{

	vector<string> arguments;

	// First argument is reserved for the name of the executable
	for(int i = 0; i < argc; ++i)
	{
		arguments.push_back(string(argv[i]));
	}
	return arguments;
}

void NonOverlapingDetections(const vector<CLMTracker::CLM>& clm_models, vector<Rect_<double> >& face_detections)
{

    // Go over the model and eliminate detections that are not informative (there already is a tracker there)
    for(size_t model = 0; model < clm_models.size(); ++model)
    {

        // See if the detections intersect
        Rect_<double> model_rect = clm_models[model].GetBoundingBox();
        
        for(int detection = face_detections.size()-1; detection >=0; --detection)
        {
            double intersection_area = (model_rect & face_detections[detection]).area();
            double union_area = model_rect.area() + face_detections[detection].area() - 2 * intersection_area;

            // If the model is already tracking what we're detecting ignore the detection, this is determined by amount of overlap
            if( intersection_area/union_area > 0.5)
            {
                face_detections.erase(face_detections.begin() + detection);
            }
        }
    }
}

// Useful utility for creating directories for storing the output files
void create_directory_from_file(string output_path)
{

	// Creating the right directory structure
	
	// First get rid of the file
	auto p = path(path(output_path).parent_path());

	if(!p.empty() && !boost::filesystem::exists(p))		
	{
		bool success = boost::filesystem::create_directories(p);
		if(!success)
		{
			cout << "Failed to create a directory... " << p.string() << endl;
		}
	}
}

void create_directory(string output_path)
{

	// Creating the right directory structure
	auto p = path(output_path);

	if(!boost::filesystem::exists(p))		
	{
		bool success = boost::filesystem::create_directories(p);
		
		if(!success)
		{
			cout << "Failed to create a directory..." << p.string() << endl;
		}
	}
}

// Extracting the following command line arguments -f, -fd, -op, -of, -ov (and possible ordered repetitions)
void get_output_feature_params(vector<string> &output_similarity_aligned, bool &vid_output, vector<string> &output_gaze_files, vector<string> &output_hog_aligned_files, vector<string> &output_model_param_files, vector<string> &output_au_files, double &similarity_scale, int &similarity_size, bool &grayscale, bool &rigid, bool& verbose, vector<string> &arguments)
{
	output_similarity_aligned.clear();
	vid_output = false;
	output_hog_aligned_files.clear();
	output_model_param_files.clear();

	bool* valid = new bool[arguments.size()];

	for(size_t i = 0; i < arguments.size(); ++i)
	{
		valid[i] = true;
	}

	string input_root = "";
	string output_root = "";

	// First check if there is a root argument (so that videos and outputs could be defined more easilly)
	for(size_t i = 0; i < arguments.size(); ++i)
	{
		if (arguments[i].compare("-root") == 0) 
		{                    
			input_root = arguments[i + 1];
			output_root = arguments[i + 1];
			i++;
		}
		if (arguments[i].compare("-inroot") == 0) 
		{                    
			input_root = arguments[i + 1];
			i++;
		}
		if (arguments[i].compare("-outroot") == 0) 
		{                    
			output_root = arguments[i + 1];
			i++;
		}
	}

	for(size_t i = 0; i < arguments.size(); ++i)
	{
		if (arguments[i].compare("-simalignvid") == 0) 
		{                    
			output_similarity_aligned.push_back(output_root + arguments[i + 1]);
			create_directory_from_file(output_root + arguments[i + 1]);
			vid_output = true;
			valid[i] = false;
			valid[i+1] = false;			
			i++;
		}		
		else if (arguments[i].compare("-oaus") == 0) 
		{                    
			output_au_files.push_back(output_root + arguments[i + 1]);
			create_directory_from_file(output_root + arguments[i + 1]);
			vid_output = true;
			valid[i] = false;
			valid[i+1] = false;			
			i++;
		}	
		else if (arguments[i].compare("-ogaze") == 0)
		{
			output_gaze_files.push_back(output_root + arguments[i + 1]);
			create_directory_from_file(output_root + arguments[i + 1]);
			valid[i] = false;
			valid[i + 1] = false;
			i++;
		}
		else if (arguments[i].compare("-simaligndir") == 0) 
		{                    
			output_similarity_aligned.push_back(output_root + arguments[i + 1]);
			create_directory(output_root + arguments[i + 1]);
			vid_output = false;
			valid[i] = false;
			valid[i+1] = false;			
			i++;
		}		
		else if(arguments[i].compare("-hogalign") == 0) 
		{
			output_hog_aligned_files.push_back(output_root + arguments[i + 1]);
			create_directory_from_file(output_root + arguments[i + 1]);
			valid[i] = false;
			valid[i+1] = false;			
			i++;
		}
		else if(arguments[i].compare("-verbose") == 0) 
		{
			verbose = true;
		}
		else if(arguments[i].compare("-oparams") == 0) 
		{
			output_model_param_files.push_back(output_root + arguments[i + 1]);
			create_directory_from_file(output_root + arguments[i + 1]);
			valid[i] = false;
			valid[i+1] = false;			
			i++;
		}
		else if(arguments[i].compare("-rigid") == 0) 
		{
			rigid = true;
		}
		else if(arguments[i].compare("-g") == 0) 
		{
			grayscale = true;
			valid[i] = false;
		}
		else if (arguments[i].compare("-simscale") == 0) 
		{                    
			similarity_scale = stod(arguments[i + 1]);
			valid[i] = false;
			valid[i+1] = false;			
			i++;
		}		
		else if (arguments[i].compare("-simsize") == 0) 
		{                    
			similarity_size = stoi(arguments[i + 1]);
			valid[i] = false;
			valid[i+1] = false;			
			i++;
		}		
		else if (arguments[i].compare("-help") == 0)
		{
			cout << "Output features are defined as: -simalign <outputfile>\n"; // Inform the user of how to use the program				
		}
	}

	for(int i=arguments.size()-1; i >= 0; --i)
	{
		if(!valid[i])
		{
			arguments.erase(arguments.begin()+i);
		}
	}

}

// Can process images via directories creating a separate output file per directory
void get_image_input_output_params_feats(vector<vector<string> > &input_image_files, bool& as_video, vector<string> &arguments)
{
	bool* valid = new bool[arguments.size()];
		
	for(size_t i = 0; i < arguments.size(); ++i)
	{
		valid[i] = true;
		if (arguments[i].compare("-fdir") == 0) 
		{                    

			// parse the -fdir directory by reading in all of the .png and .jpg files in it
			path image_directory (arguments[i+1]); 

			try
			{
				 // does the file exist and is it a directory
				if (exists(image_directory) && is_directory(image_directory))   
				{
					
					vector<path> file_in_directory;                                
					copy(directory_iterator(image_directory), directory_iterator(), back_inserter(file_in_directory));

					// Sort the images in the directory first
					sort(file_in_directory.begin(), file_in_directory.end()); 

					vector<string> curr_dir_files;

					for (vector<path>::const_iterator file_iterator (file_in_directory.begin()); file_iterator != file_in_directory.end(); ++file_iterator)
					{
						// Possible image extension .jpg and .png
						if(file_iterator->extension().string().compare(".jpg") == 0 || file_iterator->extension().string().compare(".png") == 0)
						{																
							curr_dir_files.push_back(file_iterator->string());															
						}
					}

					input_image_files.push_back(curr_dir_files);
				}
			}
			catch (const filesystem_error& ex)
			{
				cout << ex.what() << '\n';
			}

			valid[i] = false;
			valid[i+1] = false;		
			i++;
		}
		else if (arguments[i].compare("-asvid") == 0) 
		{
			as_video = true;
		}
		else if (arguments[i].compare("-help") == 0)
		{
			cout << "Input output files are defined as: -fdir <image directory (can have multiple ones)> -asvid <the images in a folder are assumed to come from a video (consecutive)>" << endl; // Inform the user of how to use the program				
		}
	}
	
	// Clear up the argument list
	for(int i=arguments.size()-1; i >= 0; --i)
	{
		if(!valid[i])
		{
			arguments.erase(arguments.begin()+i);
		}
	}

}

void output_HOG_frame(std::ofstream* hog_file, bool good_frame, const Mat_<double>& hog_descriptor, int num_rows, int num_cols)
{

	// Using FHOGs, hence 31 channels
	int num_channels = 31;

	hog_file->write((char*)(&num_cols), 4);
	hog_file->write((char*)(&num_rows), 4);
	hog_file->write((char*)(&num_channels), 4);

	// Not the best way to store a bool, but will be much easier to read it
	float good_frame_float;
	if(good_frame)
		good_frame_float = 1;
	else
		good_frame_float = -1;

	hog_file->write((char*)(&good_frame_float), 4);

	cv::MatConstIterator_<double> descriptor_it = hog_descriptor.begin();

	for(int y = 0; y < num_cols; ++y)
	{
		for(int x = 0; x < num_rows; ++x)
		{
			for(unsigned int o = 0; o < 31; ++o)
			{

				float hog_data = (float)(*descriptor_it++);
				hog_file->write ((char*)&hog_data, 4);
			}
		}
	}
}

// Some globals for tracking timing information for visualisation
double fps_tracker = -1.0;
int64 t0 = 0;

// Visualising the results
void visualise_tracking(Mat& captured_image, const CLMTracker::CLM& clm_model, const CLMTracker::CLMParameters& clm_parameters, Point3f gazeDirection0, Point3f gazeDirection1, int frame_count, double fx, double fy, double cx, double cy)
{

	// Drawing the facial landmarks on the face and the bounding box around it if tracking is successful and initialised
	double detection_certainty = clm_model.detection_certainty;
	bool detection_success = clm_model.detection_success;

	double visualisation_boundary = 0.2;

	// Only draw if the reliability is reasonable, the value is slightly ad-hoc
	if (detection_certainty < visualisation_boundary)
	{
		CLMTracker::Draw(captured_image, clm_model);

		double vis_certainty = detection_certainty;
		if (vis_certainty > 1)
			vis_certainty = 1;
		if (vis_certainty < -1)
			vis_certainty = -1;

		vis_certainty = (vis_certainty + 1) / (visualisation_boundary + 1);

		// A rough heuristic for box around the face width
		int thickness = (int)std::ceil(2.0* ((double)captured_image.cols) / 640.0);

		Vec6d pose_estimate_to_draw = CLMTracker::GetCorrectedPoseCameraPlane(clm_model, fx, fy, cx, cy);

		// Draw it in reddish if uncertain, blueish if certain
		CLMTracker::DrawBox(captured_image, pose_estimate_to_draw, Scalar((1 - vis_certainty)*255.0, 0, vis_certainty * 255), thickness, fx, fy, cx, cy);

		if (clm_parameters.track_gaze && detection_success)
		{
			FaceAnalysis::DrawGaze(captured_image, clm_model, gazeDirection0, gazeDirection1, fx, fy, cx, cy);
		}
	}

	// Work out the framerate
	if (frame_count % 10 == 0)
	{
		double t1 = cv::getTickCount();
		fps_tracker = 10.0 / (double(t1 - t0) / cv::getTickFrequency());
		t0 = t1;
	}

	// Write out the framerate on the image before displaying it
	char fpsC[255];
	std::sprintf(fpsC, "%d", (int)fps_tracker);
	string fpsSt("FPS:");
	fpsSt += fpsC;
	cv::putText(captured_image, fpsSt, cv::Point(10, 20), CV_FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(255, 0, 0));

	if (!clm_parameters.quiet_mode)
	{
		namedWindow("tracking_result", 1);
		imshow("tracking_result", captured_image);
	}
}

int main (int argc, char **argv)
{
	typedef clm_bridge::ClmHeads ClmHeadsMsg;
	typedef clm_bridge::ClmHead ClmHeadMsg;
	typedef clm_bridge::ClmEyeGaze ClmEyeGazeMsg;
	typedef clm_bridge::ClmFacialActionUnit ClmFacialActionUnitMsg;

	ros::init(argc, argv, "clm_bridge");
    ros::NodeHandle nh_rel( "~" );

   	ros::Publisher clm_heads_pub = nh_rel.advertise<ClmHeadsMsg>("heads", 10);

	vector<string> arguments = get_arguments(argc, argv);

	// Some initial parameters that can be overriden from command line	
	vector<string> files, depth_directories, pose_output_files, tracked_videos_output, landmark_output_files, landmark_3D_output_files;
	
	// By default try webcam 0
	int device = 0;


	CLMTracker::CLMParameters clm_parameters(arguments);
		
	// TODO a command line argument
	clm_parameters.track_gaze = true;

	// Get the input output file parameters
	
	// Indicates that rotation should be with respect to camera plane or with respect to camera
	bool use_camera_plane_pose;
	CLMTracker::get_video_input_output_params(files, depth_directories, pose_output_files, tracked_videos_output, landmark_output_files, landmark_3D_output_files, use_camera_plane_pose, arguments);

	bool video_input = true;
	bool verbose = true;
	bool images_as_video = false;
	bool webcam = false;

	vector<vector<string> > input_image_files;

	// Adding image support for reading in the files
	if(files.empty())
	{
		vector<string> d_files;
		vector<string> o_img;
		vector<Rect_<double>> bboxes;
		get_image_input_output_params_feats(input_image_files, images_as_video, arguments);	

		if(!input_image_files.empty())
		{
			video_input = false;
		}

	}

	// Grab camera parameters, if they are not defined (approximate values will be used)
	float fx = 0, fy = 0, cx = 0, cy = 0;
	// Get camera parameters
	CLMTracker::get_camera_params(device, fx, fy, cx, cy, arguments);    
	
	// If cx (optical axis centre) is undefined will use the image size/2 as an estimate
	bool cx_undefined = false;
	bool fx_undefined = false;
	if (cx == 0 || cy == 0)
	{
		cx_undefined = true;
	}
	if (fx == 0 || fy == 0)
	{
		fx_undefined = true;
	}

	// The modules that are being used for tracking
	CLMTracker::CLM clm_model(clm_parameters.model_location);	

	vector<string> output_similarity_align;
	vector<string> output_au_files;
	vector<string> output_hog_align_files;
	vector<string> params_output_files;
	vector<string> gaze_output_files;

	double sim_scale = 0.7;
	int sim_size = 112;
	bool grayscale = false;	
	bool video_output = false;
	bool rigid = false;	
	int num_hog_rows;
	int num_hog_cols;

	get_output_feature_params(output_similarity_align, video_output, gaze_output_files, output_hog_align_files, params_output_files, output_au_files, sim_scale, sim_size, grayscale, rigid, verbose, arguments);
	
	// Used for image masking

	Mat_<int> triangulation;
	string tri_loc;
	if(boost::filesystem::exists(path("model/tris_68_full.txt")))
	{
		std::ifstream triangulation_file("model/tris_68_full.txt");
		CLMTracker::ReadMat(triangulation_file, triangulation);
		tri_loc = "model/tris_68_full.txt";
	}
	else
	{
		path loc = path(arguments[0]).parent_path() / "model/tris_68_full.txt";
		tri_loc = loc.string();

		if(exists(loc))
		{
			std::ifstream triangulation_file(loc.string());
			CLMTracker::ReadMat(triangulation_file, triangulation);
		}
		else
		{
			cout << "Can't find triangulation files, exiting" << endl;
			return 0;
		}
	}	

	// Will warp to scaled mean shape
	Mat_<double> similarity_normalised_shape = clm_model.pdm.mean_shape * sim_scale;
	// Discard the z component
	similarity_normalised_shape = similarity_normalised_shape(Rect(0, 0, 1, 2*similarity_normalised_shape.rows/3)).clone();

	// If multiple video files are tracked, use this to indicate if we are done
	bool done = false;	
	int f_n = -1;
	int curr_img = -1;



	string au_loc;
	if(boost::filesystem::exists(path("AU_predictors/AU_all_best.txt")))
	{
		au_loc = "AU_predictors/AU_all_best.txt";
	}
	else
	{
		path loc = path(arguments[0]).parent_path() / "AU_predictors/AU_all_best.txt";

		if(exists(loc))
		{
			au_loc = loc.string();
		}
		else
		{
			cout << "Can't find AU prediction files, exiting" << endl;
			return 0;
		}
	}	

	// Creating a  face analyser that will be used for AU extraction
	FaceAnalysis::FaceAnalyser face_analyser(vector<Vec3d>(), 0.7, 112, 112, au_loc, tri_loc);
		
	while(!done) // this is not a for loop as we might also be reading from a webcam
	{
		
		string current_file;
		
		VideoCapture video_capture;
		
		Mat captured_image;
		int total_frames = -1;
		int reported_completion = 0;

		double fps_vid_in = -1.0;

		if(video_input)
		{
			// We might specify multiple video files as arguments
			if(files.size() > 0)
			{
				f_n++;			
				current_file = files[f_n];
			}
			else
			{
				// If we want to write out from webcam
				f_n = 0;
			}
			// Do some grabbing
			if( current_file.size() > 0 )
			{
				INFO_STREAM( "Attempting to read from file: " << current_file );
				video_capture = VideoCapture( current_file );
				total_frames = (int)video_capture.get(CV_CAP_PROP_FRAME_COUNT);
				fps_vid_in = video_capture.get(CV_CAP_PROP_FPS);

				// Check if fps is nan or less than 0
				if (fps_vid_in != fps_vid_in || fps_vid_in <= 0)
				{
					INFO_STREAM("FPS of the video file cannot be determined, assuming 30");
					fps_vid_in = 30;
				}
			}
			else
			{
				INFO_STREAM( "Attempting to capture from device: " << device );
				video_capture = VideoCapture( device );
				webcam = true;

				// Read a first frame often empty in camera
				Mat captured_image;
				video_capture >> captured_image;
			}

			if( !video_capture.isOpened() ) FATAL_STREAM( "Failed to open video source" );
			else INFO_STREAM( "Device or file opened");

			video_capture >> captured_image;	
		}
		else
		{
			f_n++;	
			curr_img++;
			if(!input_image_files[f_n].empty())
			{
				string curr_img_file = input_image_files[f_n][curr_img];
				captured_image = imread(curr_img_file, -1);
			}
			else
			{
				FATAL_STREAM( "No .jpg or .png images in a specified drectory" );
			}

		}	
		
		// If optical centers are not defined just use center of image
		if(cx_undefined)
		{
			cx = captured_image.cols / 2.0f;
			cy = captured_image.rows / 2.0f;
		}
		// Use a rough guess-timate of focal length
		if (fx_undefined)
		{
			fx = 500 * (captured_image.cols / 640.0);
			fy = 500 * (captured_image.rows / 480.0);

			fx = (fx + fy) / 2.0;
			fy = fx;
		}
	
		// Creating output files
		std::ofstream gaze_output_file;
		if (!gaze_output_files.empty())
		{
			gaze_output_file.open(gaze_output_files[f_n], ios_base::out);

			gaze_output_file << "frame, timestamp, confidence, success, x_0, y_0, z_0, x_1, y_1, z_1, x_h0, y_h0, z_h0, x_h1, y_h1, z_h1";
			gaze_output_file << endl;
		}

		std::ofstream pose_output_file;
		if(!pose_output_files.empty())
		{
			pose_output_file.open (pose_output_files[f_n], ios_base::out);

			pose_output_file << "frame, timestamp, confidence, success, Tx, Ty, Tz, Rx, Ry, Rz";
			pose_output_file << endl;
		}
	
		std::ofstream landmarks_output_file;		
		if(!landmark_output_files.empty())
		{
			landmarks_output_file.open(landmark_output_files[f_n], ios_base::out);

			landmarks_output_file << "frame, timestamp, confidence, success";
			for(int i = 0; i < clm_model.pdm.NumberOfPoints(); ++i)
			{

				landmarks_output_file << ", x" << i;
			}
			for(int i = 0; i < clm_model.pdm.NumberOfPoints(); ++i)
			{

				landmarks_output_file << ", y" << i;
			}
			landmarks_output_file << endl;
		}

		std::ofstream landmarks_3D_output_file;		
		if(!landmark_3D_output_files.empty())
		{
			landmarks_3D_output_file.open(landmark_3D_output_files[f_n], ios_base::out);

			landmarks_3D_output_file << "frame, timestamp, confidence, success";
			for(int i = 0; i < clm_model.pdm.NumberOfPoints(); ++i)
			{

				landmarks_3D_output_file << ", X" << i;
			}
			for(int i = 0; i < clm_model.pdm.NumberOfPoints(); ++i)
			{

				landmarks_3D_output_file << ", Y" << i;
			}
			for(int i = 0; i < clm_model.pdm.NumberOfPoints(); ++i)
			{

				landmarks_3D_output_file << ", Z" << i;
			}
			landmarks_3D_output_file << endl;
		}

		// Outputting model parameters (rigid and non-rigid), the first parameters are the 6 rigid shape parameters, they are followed by the non rigid shape parameters
		std::ofstream params_output_file;		
		if(!params_output_files.empty())
		{
			params_output_file.open(params_output_files[f_n], ios_base::out);

			params_output_file << "frame, timestamp, confidence, success, scale, rx, ry, rz, tx, ty";
			for(int i = 0; i < clm_model.pdm.NumberOfModes(); ++i)
			{

				params_output_file << ", p" << i;
			}
			params_output_file << endl;

		}

		std::ofstream au_output_file;
		if(!output_au_files.empty())
		{
			au_output_file.open (output_au_files[f_n], ios_base::out);

			vector<string> au_names_class = face_analyser.GetAUClassNames();
			vector<string> au_names_reg = face_analyser.GetAURegNames();

			au_output_file << "frame, timestamp, confidence, success";
			for(string reg_name : au_names_reg)
			{
				au_output_file << ", " << reg_name << "_r";
			}
			
			for(string class_name : au_names_class)
			{
				au_output_file << ", " << class_name << "_c";
			}
			au_output_file << endl;

		}

		// saving the videos
		VideoWriter output_similarity_aligned_video;
		if(!output_similarity_align.empty())
		{
			if(video_output)
			{
				double fps = webcam ? 30 : fps_vid_in;
				output_similarity_aligned_video = VideoWriter(output_similarity_align[f_n], CV_FOURCC('H', 'F', 'Y', 'U'), fps, Size(sim_size, sim_size), true);
			}
		}
		
		// Saving the HOG features
		std::ofstream hog_output_file;
		if(!output_hog_align_files.empty())
		{
			hog_output_file.open(output_hog_align_files[f_n], ios_base::out | ios_base::binary);
		}

		// saving the videos
		VideoWriter writerFace;
		if(!tracked_videos_output.empty())
		{
			double fps = webcam ? 30 : fps_vid_in;
			writerFace = VideoWriter(tracked_videos_output[f_n], CV_FOURCC('D', 'I', 'V', 'X'), fps, captured_image.size(), true);
		}

		int frame_count = 0;
		
		// This is useful for a second pass run (if want AU predictions)
		vector<Vec6d> params_global_video;
		vector<bool> successes_video;
		vector<Mat_<double>> params_local_video;
		vector<Mat_<double>> detected_landmarks_video;
				
		// Use for timestamping if using a webcam
		int64 t_initial = cv::getTickCount();

		bool visualise_hog = verbose;

		// Timestamp in seconds of current processing
		double time_stamp = 0;

		INFO_STREAM( "Starting tracking");
		while( (!captured_image.empty()) && nh_rel.ok() )
		{		
			// Grab the timestamp first
			if (webcam)
			{
				int64 curr_time = cv::getTickCount();
				time_stamp = (double(curr_time - t_initial) / cv::getTickFrequency());
			}
			else if (video_input)
			{
				time_stamp = (double)frame_count * (1.0 / fps_vid_in);				
			}
			else
			{
				time_stamp = 0.0;
			}

			// Reading the images
			Mat_<uchar> grayscale_image;

			if(captured_image.channels() == 3)
			{
				cvtColor(captured_image, grayscale_image, CV_BGR2GRAY);				
			}
			else
			{
				grayscale_image = captured_image.clone();				
			}
		
			// The actual facial landmark detection / tracking
			bool detection_success;
			
			if(video_input || images_as_video)
			{
				detection_success = CLMTracker::DetectLandmarksInVideo(grayscale_image, clm_model, clm_parameters);
			}
			else
			{
				detection_success = CLMTracker::DetectLandmarksInImage(grayscale_image, clm_model, clm_parameters);
			}
			
			// Gaze tracking, absolute gaze direction
			Point3f gazeDirection0;
			Point3f gazeDirection1;

			// Gaze with respect to head rather than camera (for example if eyes are rolled up and the head is tilted or turned this will be stable)
			Point3f gazeDirection0_head;
			Point3f gazeDirection1_head;

			if (clm_parameters.track_gaze && detection_success)
			{
				FaceAnalysis::EstimateGaze(clm_model, gazeDirection0, gazeDirection0_head, fx, fy, cx, cy, true);
				FaceAnalysis::EstimateGaze(clm_model, gazeDirection1, gazeDirection1_head, fx, fy, cx, cy, false);
			}

			// Do face alignment
			Mat sim_warped_img;			
			Mat_<double> hog_descriptor;

			// But only if needed in output
			if(!output_similarity_align.empty() || hog_output_file.is_open() || !output_au_files.empty())
			{
				face_analyser.AddNextFrame(captured_image, clm_model, time_stamp, webcam, !clm_parameters.quiet_mode);
				face_analyser.GetLatestAlignedFace(sim_warped_img);

				//FaceAnalysis::AlignFaceMask(sim_warped_img, captured_image, clm_model, triangulation, rigid, sim_scale, sim_size, sim_size);			
				if(!clm_parameters.quiet_mode)
				{
					cv::imshow("sim_warp", sim_warped_img);			
				}
				if(hog_output_file.is_open())
				{
					FaceAnalysis::Extract_FHOG_descriptor(hog_descriptor, sim_warped_img, num_hog_rows, num_hog_cols);						

					if(visualise_hog && !clm_parameters.quiet_mode)
					{
						Mat_<double> hog_descriptor_vis;
						FaceAnalysis::Visualise_FHOG(hog_descriptor, num_hog_rows, num_hog_cols, hog_descriptor_vis);
						cv::imshow("hog", hog_descriptor_vis);	
					}
				}
			}

			// Work out the pose of the head from the tracked model
			Vec6d pose_estimate_CLM;
			if(use_camera_plane_pose)
			{
				pose_estimate_CLM = CLMTracker::GetCorrectedPoseCameraPlane(clm_model, fx, fy, cx, cy);
			}
			else
			{
				pose_estimate_CLM = CLMTracker::GetCorrectedPoseCamera(clm_model, fx, fy, cx, cy);
			}

			if(hog_output_file.is_open())
			{
				output_HOG_frame(&hog_output_file, detection_success, hog_descriptor, num_hog_rows, num_hog_cols);
			}

			// Write the similarity normalised output
			if(!output_similarity_align.empty())
			{
				if(video_output)
				{
					if(output_similarity_aligned_video.isOpened())
					{
						output_similarity_aligned_video << sim_warped_img;
					}
				}
				else
				{
					char name[100];
					
					// output the frame number
					std::sprintf(name, "frame_det_%06d.png", frame_count);

					// Construct the output filename
					boost::filesystem::path slash("/");
					
					std::string preferredSlash = slash.make_preferred().string();
				
					string out_file = output_similarity_align[f_n] + preferredSlash + string(name);
					imwrite(out_file, sim_warped_img);
				}
			}

			// Visualising the tracker
			visualise_tracking(captured_image, clm_model, clm_parameters, gazeDirection0, gazeDirection1, frame_count, fx, fy, cx, cy);

			//output the tracked video
			if(!tracked_videos_output.empty())
			{		
				writerFace << captured_image;
			}

			if(video_input)
			{
				video_capture >> captured_image;
			}
			else
			{
				curr_img++;
				if(curr_img < (int)input_image_files[f_n].size())
				{
					string curr_img_file = input_image_files[f_n][curr_img];
					captured_image = imread(curr_img_file, -1);
				}
				else
				{
					captured_image = Mat();
				}
			}
			// detect key presses
			char character_press = cv::waitKey(1);
			
			// restart the tracker
			if(character_press == 'r')
			{
				clm_model.Reset();
			}
			// quit the application
			else if(character_press=='q')
			{
				return(0);
			}

			// Update the frame count
			frame_count++;

			if(total_frames != -1)
			{
				if((double)frame_count/(double)total_frames >= reported_completion / 10.0)
				{
					cout << reported_completion * 10 << "% ";
					reported_completion = reported_completion + 1;
				}
			}

			double confidence = 0.5 * (1 - clm_model.detection_certainty);



			//TODO: Add multiple head-tracking feature
			ClmHeadsMsg ros_heads_msg;

			ClmHeadMsg ros_head_msg;
			auto & ros_eyegazes_msg = ros_head_msg.eyegazes;
			auto & ros_aus_msg = ros_head_msg.aus;

			ros_head_msg.detection_success = static_cast<uint8_t>( detection_success );
			ros_head_msg.detection_confidence = static_cast<float>( confidence );
			ros_head_msg.time_stamp = static_cast<float>( time_stamp );

			// package head pose message
			ros_head_msg.headpose.x = static_cast<float>( pose_estimate_CLM[0] );
			ros_head_msg.headpose.y = static_cast<float>( pose_estimate_CLM[1] );
			ros_head_msg.headpose.z = static_cast<float>( pose_estimate_CLM[2] );
			ros_head_msg.headpose.pitch = static_cast<float>( pose_estimate_CLM[3] );
			ros_head_msg.headpose.yaw = static_cast<float>( pose_estimate_CLM[4] );
			ros_head_msg.headpose.roll = static_cast<float>( pose_estimate_CLM[5] );


			std::vector<Point3f> gazeDirections = {gazeDirection0, gazeDirection1};
			std::vector<Point3f> gazeDirections_head = {gazeDirection0_head, gazeDirection1_head};

			for (size_t p = 0; p < gazeDirections_head.size(); p++)
			{
				ClmEyeGazeMsg ros_eyegaze_msg;
				ros_eyegaze_msg.eye_id = p;
				ros_eyegaze_msg.gaze_direction_cameraref_x = static_cast<float>( gazeDirections[p].x ); 
				ros_eyegaze_msg.gaze_direction_cameraref_y = static_cast<float>( gazeDirections[p].y );
				ros_eyegaze_msg.gaze_direction_cameraref_z = static_cast<float>( gazeDirections[p].z );
				ros_eyegaze_msg.gaze_direction_headref_x = static_cast<float>( gazeDirections_head[p].x ); //lateral gaze
				ros_eyegaze_msg.gaze_direction_headref_y = static_cast<float>( gazeDirections_head[p].y );
				ros_eyegaze_msg.gaze_direction_headref_z = static_cast<float>( gazeDirections_head[p].z );
				ros_eyegazes_msg.emplace_back( std::move( ros_eyegaze_msg ) );
			}

			//AU01_r, AU04_r, AU06_r, AU10_r, AU12_r, AU14_r, AU17_r, AU25_r, AU02_r, AU05_r, AU09_r, AU15_r, AU20_r, AU26_r, AU12_c, AU23_c, AU28_c, AU04_c, AU15_c, AU45_c

			// package facial action unit message
			std::vector<int> au_r_handles = {1, 4, 6, 10, 12, 14, 17, 25, 2, 5, 9, 15, 20, 26};
			std::vector<int> au_c_handles = {12, 23, 28, 4, 15, 45};

			auto aus_reg = face_analyser.GetCurrentAUsReg();
			
			if(aus_reg.size() == 0)
			{
				for(size_t p = 0; p < face_analyser.GetAURegNames().size(); p++)
				{
					ClmFacialActionUnitMsg ros_au_msg;
					ros_au_msg.au_type = static_cast<uint8_t>( au_r_handles[p] );
					ros_au_msg.au_value = 0;
					ros_au_msg.au_prediction_method = 0;
					ros_aus_msg.emplace_back( std::move( ros_au_msg ) );
				}
			}
			else
			{
				for(size_t p = 0; p < aus_reg.size(); p++)
				{
					ClmFacialActionUnitMsg ros_au_msg;
					ros_au_msg.au_type = static_cast<uint8_t>( au_r_handles[p] );
					ros_au_msg.au_value = static_cast<float>( aus_reg[p].second );
					ros_au_msg.au_prediction_method = 0;
					ros_aus_msg.emplace_back( std::move( ros_au_msg ) );
				}
			}

			auto aus_class = face_analyser.GetCurrentAUsClass();
			
			if(aus_class.size() == 0)
			{
				for(size_t p = 0; p < face_analyser.GetAUClassNames().size(); p++)
				{
					ClmFacialActionUnitMsg ros_au_msg;
					ros_au_msg.au_type = static_cast<uint8_t>( au_c_handles[p] );
					ros_au_msg.au_value = 0;
					ros_au_msg.au_prediction_method = 1;
					ros_aus_msg.emplace_back( std::move( ros_au_msg ) );
				}
			}
			else
			{
				for(size_t p = 0; p < aus_class.size(); p++)
				{
					ClmFacialActionUnitMsg ros_au_msg;
					ros_au_msg.au_type = static_cast<uint8_t>( au_c_handles[p] );
					ros_au_msg.au_value = static_cast<float>( aus_class[p].second );
					ros_au_msg.au_prediction_method = 1;
					ros_aus_msg.emplace_back( std::move( ros_au_msg ) );
				}
			}

			ros_heads_msg.heads.emplace_back( std::move( ros_head_msg ) );
			clm_heads_pub.publish( ros_heads_msg );
			ros::spinOnce();
		}
		
		if(total_frames != -1)
		{
			cout << endl;
		}

		frame_count = 0;
		curr_img = -1;

		// Reset the model, for the next video
		clm_model.Reset();

		pose_output_file.close();
		gaze_output_file.close();
		landmarks_output_file.close();

		vector<double> certainties;
		vector<bool> successes;
		vector<double> timestamps;
		vector<std::pair<std::string, vector<double>>> predictions_reg;
		vector<std::pair<std::string, vector<double>>> predictions_class;

		face_analyser.ExtractAllPredictionsOfflineReg(predictions_reg, certainties, successes, timestamps);
		face_analyser.ExtractAllPredictionsOfflineClass(predictions_class, certainties, successes, timestamps);

		// Output all of the AU stuff with offline correction and cleanup
		if(!output_au_files.empty())
		{			

			au_output_file.close();

			au_output_file.open (output_au_files[f_n], ios_base::out);

			au_output_file << "frame, timestamp, confidence, success";
			for(auto reg_name : predictions_reg)
			{
				au_output_file << ", " << reg_name.first << "_r";
			}
			
			for(auto class_name : predictions_class)
			{
				au_output_file << ", " << class_name.first << "_c";
			}
			au_output_file << endl;

			for(size_t frame = 0; frame < certainties.size(); ++frame)
			{
				double detection_certainty = certainties[frame];
				bool detection_success = successes[frame];

				double confidence = 0.5 * (1 - detection_certainty);

				au_output_file << frame + 1 << ", " << timestamps[frame] << ", " << confidence << ", " << detection_success;
				auto aus_reg = face_analyser.GetCurrentAUsReg();
				
				for(auto au_reg : predictions_reg)
				{
					au_output_file << ", " << au_reg.second[frame];
				}

				if(aus_reg.size() == 0)
				{
					for(size_t p = 0; p < face_analyser.GetAURegNames().size(); ++p)
					{
						au_output_file << ", 0";
					}
				}

				auto aus_class = face_analyser.GetCurrentAUsClass();
				
				for(auto au_class : predictions_class)
				{
					au_output_file << ", " << au_class.second[frame];
				}

				if(aus_class.size() == 0)
				{
					for(size_t p = 0; p < face_analyser.GetAUClassNames().size(); ++p)
					{
						au_output_file << ", 0";
					}
				}
				au_output_file << endl;
			}
			face_analyser.Reset();
		}
		// break out of the loop if done with all the files (or using a webcam)
		if(f_n == files.size() -1 || files.empty())
		{
			done = true;
		}
	}

	return 0;
}



//how do I deal with the while loop??
//how do other things deal with the loop??

// int main( int argc, char ** argv )
// {
// 	vector<string> arguments = get_arguments(argc, argv);

//     ros::init( argc, argv, "clm_bridge" );
 
//     ros::NodeHandle nh_rel( "~" );
//     KinectBridge2Client kinect_bridge2_client( nh_rel );
 
//     process_and_publish_CLM_messages();

//     return 0;
// }
