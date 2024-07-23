// (c) 2024 Hajo Nils Krabbenhöft

// This program will load all the individual frames from a RevoScan project and then re-align them using PCL features

#include <iostream>
#include <fstream>
#include <nlohmann/json.hpp>
#include <Eigen/Core>
#include <filesystem>
#include <pcl/point_cloud.h>
#include <pcl/impl/point_types.hpp>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/registration/ia_ransac.h>
#include <pcl/registration/transformation_estimation_svd.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/search/search.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/keypoints/harris_3d.h>
#include <pcl/common/time.h>
#include <pcl/sample_consensus/sac_model_registration.h>
#include <pcl/sample_consensus/ransac.h>
#include <thread>

std::random_device rng_rd;
std::mt19937 rng(rng_rd());

// the merged_* variables contain our current globally consistent state, so this is what we align new data against
auto merged_positions = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
auto merged_normals = std::make_shared<pcl::PointCloud<pcl::Normal>>();
auto merged_signatures = std::make_shared<pcl::PointCloud<pcl::FPFHSignature33>>();
std::vector<int> merged_lifetime;

// load_file will load a depth frame,
// re-project it back into 3D with the Q matrix and scale from the RevoScan project file
// and then calculate FPFH signatures for the environment around each point
// and then it'll create a list of indices where the FPFH signature has strong curvature in at least one direction
void load_file(const std::filesystem::path& file, std::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>& positions, std::shared_ptr<pcl::PointCloud<pcl::Normal>>& normals, std::shared_ptr<pcl::PointCloud<pcl::FPFHSignature33>>& signatures, pcl::Indices& interesting_indices, Eigen::Matrix4f& Q, double depth_scale);

// calculate_hits will find those points in our globally consistent state
// that most likely match new points in our newly loaded point cloud
// based on the FPFH signatures
void calculate_hits(std::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>& hits_src, std::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>& hits_dst, const std::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>& positions, const std::shared_ptr<pcl::PointCloud<pcl::FPFHSignature33>>& signatures, const pcl::Indices& vector);

// given two arrays of matching points,
// estimate_transformation_matrix will estimate the transformation matrix
// that causes those points to overlap
bool estimate_transformation_matrix(const std::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>& hits_src, const std::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>& hits_dst, Eigen::Matrix4d& transformation_matrix);

int main() {
    std::ofstream output("output.txt", std::ofstream::out);

    std::string property_file = "~/Project03152024121357/data/c142fc59-1733-49b4-8695-a9cb78ce6df3/property.rvproj";
    std::string base_folder = "~/Project03152024121357/data/c142fc59-1733-49b4-8695-a9cb78ce6df3";

    // read the RevoScan project file and the camera's extrinsic calibration

    double depth_scale;
    {
        std::ifstream prop_f(property_file);
        auto data = nlohmann::json::parse(prop_f);
        depth_scale = data["scan_param"]["depth_scale"];
    }

    Eigen::Matrix4f Q;
    {
        std::ifstream in(base_folder + "/param/Q.bin", std::ios::in | std::ios::binary);
        assert(in.is_open());
        in.read((char*)Q.data(), sizeof(float) * Q.size());
        assert(!in.fail());
    }
    std::cout << Q << std::endl;

    // find all depth files

    std::vector<std::filesystem::path> depth_files;
    for(const auto &file : std::filesystem::directory_iterator(base_folder+"/cache")) {
        if(file.path().extension() != ".dph") continue;
        depth_files.emplace_back(file.path());
    }
    std::sort(depth_files.begin(), depth_files.end());



    while(!depth_files.empty()) {
        unsigned long fidx = rng() % depth_files.size();
        auto const& file = depth_files[fidx];

        auto positions = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
        auto normals = std::make_shared<pcl::PointCloud<pcl::Normal>>();
        auto signatures = std::make_shared<pcl::PointCloud<pcl::FPFHSignature33>>();
        pcl::Indices interesting_indices;

        load_file(file, positions, normals, signatures, interesting_indices, Q, depth_scale);

        if(false) {
            pcl::visualization::PCLVisualizer viewer("Simple Cloud Viewer");
            viewer.addPointCloudNormals<pcl::PointXYZ, pcl::Normal> (positions, normals, 10, 1.0, "normals");
            viewer.initCameraParameters ();
            viewer.resetCamera();
            while (!viewer.wasStopped ())
            {
                viewer.spinOnce (100);
                std::this_thread::sleep_for(std::chrono::nanoseconds(100));
            }
        }

        if(merged_positions->empty()) {
            // if we have no global state, this is the new reference point

            Eigen::Matrix4d ident;
            ident.setIdentity();
            output << file << std::endl << ident << std::endl;

            pcl::copyPointCloud(*positions, interesting_indices, *merged_positions);
            pcl::copyPointCloud(*normals, interesting_indices, *merged_normals);
            pcl::copyPointCloud(*signatures, interesting_indices, *merged_signatures);
            for(auto i : interesting_indices) merged_lifetime.emplace_back(10);
            std::cout << "FIRST: " << merged_positions->size() << std::endl;
            continue;
        }

        // can we recognize any shapes in this newly scanned frame
        // as being the same as known globally aligned shapes?

        auto hits_src = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
        auto hits_dst = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
        calculate_hits(hits_src, hits_dst, positions, signatures, interesting_indices);
        if(hits_src->size() < 500) continue;

        // use those hits to align the newly loaded frame into our global state

        Eigen::Matrix4d transformation_matrix;
        if(!estimate_transformation_matrix(hits_src, hits_dst, transformation_matrix))
            continue;

        // store the new transformation matrix
        output << file << std::endl << transformation_matrix << std::endl;

        auto transformed_positions = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
        pcl::transformPointCloud (*positions, *transformed_positions, transformation_matrix);

        bool show_viewer = merged_positions->size() > 10*1000*1000;
        if(show_viewer) {
            // debug visualization

            pcl::visualization::PCLVisualizer viewer("Merge View");
            pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> red(transformed_positions, 255, 0, 0);
            viewer.addPointCloud<pcl::PointXYZ>(transformed_positions, red, "new");
            pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> green(merged_positions, 0, 255, 0);
            viewer.addPointCloud<pcl::PointXYZ>(merged_positions, green, "old");
            viewer.initCameraParameters ();
            viewer.resetCamera();
            while (!viewer.wasStopped ())
            {
                viewer.spinOnce (100);
                std::this_thread::sleep_for(std::chrono::nanoseconds(100));
            }
        }

        {
            // merge new points into global state
            for(auto i : interesting_indices) merged_positions->emplace_back((*transformed_positions)[i]);
            for(auto i : interesting_indices) merged_normals->emplace_back((*normals)[i]);
            for(auto i : interesting_indices) merged_signatures->emplace_back((*signatures)[i]);
            for(auto i : interesting_indices) merged_lifetime.emplace_back(10);
            std::cout << "MERGED: " << merged_positions->size() << std::endl;
        }

        {
            // prune old points from global state
            size_t num_items = merged_lifetime.size();
            for(size_t i = 0;i<num_items;++i) {
                if( (--merged_lifetime[i]) > 0 ) continue;
                (*merged_positions)[i] = merged_positions->back();
                (*merged_normals)[i] = merged_normals->back();
                (*merged_signatures)[i] = merged_signatures->back();
                merged_lifetime[i] = merged_lifetime.back();
                --num_items;
                --i;
                merged_positions->resize(num_items);
                merged_normals->resize(num_items);
                merged_signatures->resize(num_items);
                merged_lifetime.resize(num_items);
            }
            std::cout << "PRUNED: " << merged_positions->size() << std::endl;
        }

        depth_files.erase(depth_files.begin()+fidx);
    }

    return 0;
}

bool estimate_transformation_matrix(const std::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>& hits_src, const std::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>& hits_dst, Eigen::Matrix4d& transformation_matrix) {

    // first, do a classical RANSAC to find a general alignment

    std::shared_ptr<pcl::SampleConsensusModelRegistration<pcl::PointXYZ>> reg(new pcl::SampleConsensusModelRegistration<pcl::PointXYZ>(hits_src));
    reg->setInputTarget(hits_dst);
    pcl::RandomSampleConsensus<pcl::PointXYZ> ransac(reg);
    ransac.setDistanceThreshold(0.5);
    ransac.setProbability(0.9999);
    ransac.setMaxIterations(10000);
    ransac.computeModel();

    pcl::Indices inliers;
    ransac.getInliers(inliers);
    double inlier_rate = double(inliers.size()) / double(hits_src->size());
    std::cout << "inliers: " << inliers.size() << " rate:" << inlier_rate << std::endl;
    if(inliers.size() < 200 || inlier_rate < 0.25) return false;

    Eigen::VectorXf coeff;
    ransac.getModelCoefficients(coeff);
    Eigen::Matrix4f transform;
    transform.row(0).matrix() = coeff.segment<4>(0);
    transform.row(1).matrix() = coeff.segment<4>(4);
    transform.row(2).matrix() = coeff.segment<4>(8);
    transform.row(3).matrix() = coeff.segment<4>(12);
    std::cout << transform << std::endl;

    // create arrays with only the inlier points

    Eigen::Matrix<double, 3, Eigen::Dynamic> src(3, inliers.size());
    Eigen::Matrix<double, 3, Eigen::Dynamic> tgt(3, inliers.size());

    for( std::size_t i = 0; i < inliers.size(); ++i ) {
        src(0, i) = (*hits_src)[inliers[i]].x;
        src(1, i) = (*hits_src)[inliers[i]].y;
        src(2, i) = (*hits_src)[inliers[i]].z;

        tgt(0, i) = (*hits_dst)[inliers[i]].x;
        tgt(1, i) = (*hits_dst)[inliers[i]].y;
        tgt(2, i) = (*hits_dst)[inliers[i]].z;
    }

    // umeyama to get the final transformation matrix

    transformation_matrix = pcl::umeyama(src, tgt, false);
    std::cout << transformation_matrix << std::endl;

    return true;
}

void calculate_hits(std::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>& hits_src, std::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>& hits_dst, const std::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>& positions, const std::shared_ptr<pcl::PointCloud<pcl::FPFHSignature33>>& signatures, const pcl::Indices& interesting_indices) {

    // construct KD trees to speed up searching

    pcl::search::KdTree<pcl::FPFHSignature33> sourceSignatures;
    sourceSignatures.setInputCloud(signatures);
    pcl::search::KdTree<pcl::FPFHSignature33> targetSignatures;
    targetSignatures.setInputCloud(merged_signatures);

    double sqr_dst_threshold = pow(3.0, 2.0);

    // parallel symmetrical similarity search

#pragma omp parallel for
    for(auto sidx : interesting_indices) {
        auto const& src = (*signatures)[sidx];
        pcl::Indices k_indices;
        std::vector<float> k_sqr_distances;
        if(targetSignatures.nearestKSearch(src, 1, k_indices, k_sqr_distances) > 0) {
            if(k_sqr_distances.front() < sqr_dst_threshold) {
                const auto tidx = k_indices.front();
                auto const& dst = (*merged_signatures)[tidx];
                k_indices.clear();
                k_sqr_distances.clear();
                if(sourceSignatures.nearestKSearch(dst, 1, k_indices, k_sqr_distances) > 0) {
                    if( k_sqr_distances.front() < sqr_dst_threshold ) {
                        if( k_indices.front() == sidx) {
                            #pragma omp critical
                            {
//                                    std::cout << "HIT dst:" << sqrt(k_sqr_distances.front()) << std::endl;
                                hits_src->emplace_back((*positions)[sidx]);
                                hits_dst->emplace_back((*merged_positions)[tidx]);
                                merged_lifetime[tidx] = 10;
                            }
                        }
                    }
                }
            }
        }
    }

    std::cout << "num hits:" << hits_src->size() << std::endl;
}

void load_file(const std::filesystem::path& file, std::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>& positions, std::shared_ptr<pcl::PointCloud<pcl::Normal>>& normals, std::shared_ptr<pcl::PointCloud<pcl::FPFHSignature33>>& signatures, pcl::Indices& interesting_indices, Eigen::Matrix4f& Q, double depth_scale) {
    std::cout << file << std::endl;
    std::ifstream in(file, std::ios::in | std::ios::binary);
    assert(in.is_open() && !in.fail());
    for(int y=0;y<600;++y) {
        for(int x=0;x<800;++x) {
            uint16_t rdepth;
            in.read((char*)&rdepth, sizeof(rdepth));
            assert(!in.fail());
            float depth = float(double(rdepth) * depth_scale);
            if(depth < 1e-6) continue;
            Eigen::Vector4f src(float(x),float(y),depth,1.0f);
            Eigen::Vector4f pt = Q.transpose() * src;
            pt *= depth / pt.z();
            positions->emplace_back(-pt.x(), -pt.y(), pt.z());
        }
    }

    auto search_method = std::make_shared<pcl::search::KdTree<pcl::PointXYZ>>();

    {
        pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> ne;
        ne.setViewPoint(0,0,0);
        ne.setInputCloud(positions);
//            ne.setSearchMethod(search_method);
        ne.setRadiusSearch(1.0);
        ne.compute(*normals);
    }

    std::cout << "positions:" << positions->size() << std::endl;
    {
        pcl::Indices indices;
        for (size_t i = 0; i < normals->size(); ++i)
        {
            if (!pcl::isFinite<pcl::Normal>((*normals)[i]))
                continue;
            indices.push_back(i);
        }

        auto positions2 = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
        auto normals2 = std::make_shared<pcl::PointCloud<pcl::Normal>>();
        pcl::copyPointCloud(*positions, indices, *positions2);
        pcl::copyPointCloud(*normals, indices, *normals2);
        std::swap(positions, positions2);
        std::swap(normals, normals2);
    }
    std::cout << "positions:" << positions->size() << std::endl;

    {
        pcl::FPFHEstimationOMP<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33> fest;
        fest.setInputCloud(positions);
//            fest.setSearchMethod(search_method);
//            fest.setKSearch(30);
        fest.setRadiusSearch(5.0);
        fest.setInputNormals(normals);
        fest.compute(*signatures);
    }

    {
        for (size_t i = 0; i < signatures->size(); ++i)
        {
            auto const& hist = ((*signatures)[i]).histogram;
            if( hist[5] > 5 && hist[11+5] > 5 && hist[11+11+5] > 5 )
                continue;
            interesting_indices.push_back(i);
        }
        std::cout << "interesting:" << interesting_indices.size() << std::endl;
        /*
        auto positions2 = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
        auto normals2 = std::make_shared<pcl::PointCloud<pcl::Normal>>();
        auto signatures2 = std::make_shared<pcl::PointCloud<pcl::FPFHSignature33>>();
        pcl::copyPointCloud(*positions, indices, *positions2);
        pcl::copyPointCloud(*normals, indices, *normals2);
        pcl::copyPointCloud(*signatures, indices, *signatures2);
        std::swap(positions, positions2);
        std::swap(normals, normals2);
        std::swap(signatures, signatures2);
         */
    }
}
