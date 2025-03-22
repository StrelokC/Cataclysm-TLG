#include "vehicle.h" // IWYU pragma: associated

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <queue>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "avatar.h"
#include "character.h"
#include "coordinates.h"
#include "creature_tracker.h"
#include "cuboid_rectangle.h"
#include "debug.h"
#include "enums.h"
#include "flood_fill.h"
#include "hash_utils.h"
#include "map.h"
#include "map_iterator.h"
#include "map_memory.h"
#include "mapdata.h"
#include "messages.h"
#include "point.h"
#include "tileray.h"
#include "translations.h"
#include "type_id.h"
#include "veh_type.h"
#include "vpart_position.h"

static const ter_str_id ter_t_open_air("t_open_air");

static constexpr int OMT_SIZE = coords::map_squares_per(coords::omt);
static constexpr int NAV_MAP_NUM_OMT = 2;
static constexpr int NAV_MAP_SIZE_X = NAV_MAP_NUM_OMT * OMT_SIZE;
static constexpr int NAV_MAP_SIZE_Y = OMT_SIZE;
static constexpr int NAV_VIEW_PADDING = OMT_SIZE;
static constexpr int NAV_VIEW_SIZE_X = NAV_MAP_SIZE_X + 2 * NAV_VIEW_PADDING;
static constexpr int NAV_VIEW_SIZE_Y = NAV_MAP_SIZE_Y + 2 * NAV_VIEW_PADDING;
static constexpr int TURNING_INCREMENT = 15;
static constexpr int NUM_ORIENTATIONS = 360 / TURNING_INCREMENT;
// min and max speed in tiles/s
static constexpr int MIN_SPEED_TPS = 1;
static constexpr int MAX_SPEED_TPS = 1;
static constexpr int VMIPH_PER_TPS = static_cast<int>(vehicles::vmiph_per_tile);

// Contains movement info about next navigation step
struct navigation_step
{
    navigation_step() {}

    // Vehicle direction resets after every vehicle turn
    tripoint_abs_ms position_abs;
    // Use to advance vehicle position
    tileray vehicle_direction;
    // Current angle of vehicle
    int angle = 0;

    // I assume, that distance is accumulated and after being higher
    // than VMIPH_PER_TPS, vehicle position is advanced by 1 tile
    int travelled_distance = 0;
    int cruise_velocity = 0;
    int velocity = 0;

    // If set to nullptr means it's root step(first step in a path)
    navigation_step *previous = nullptr;
    // The more expensive move_cost is the further in queue this node end up
    float move_cost = 0;
    // Represents one second of movement
    int depth = 0;

    bool operator()(navigation_step *one, navigation_step *other)
    {
        return one->move_cost > other->move_cost;
    }
};

// Contains info about vehicle collision
struct vehicle_profile
{
    vehicle_profile() {}

    // Elements, that occupy some area at given position, idk
    std::unordered_set<point> occupied_areas;
    // Areas, that will be checked for collision while moving in given direction
    std::vector<point> ray_areas;
    // Direction in which vehicle will move
    tileray ray_direction;

    void Clear()
    {
        occupied_areas.clear();
        ray_areas.clear();
    }

    void AddRayArea(point ray)
    {
        ray_areas.push_back(ray);
    }

    void AddOccupiedArea(point area_pos)
    {
        if (occupied_areas.find(area_pos) == occupied_areas.end())
            occupied_areas.emplace(area_pos);
    }

    bool IsAreaOccupied(point area_pos)
    {
        return occupied_areas.find(area_pos) != occupied_areas.end();
    }
};

// Nav info about omt
struct omt_nav_info
{
    // Okay so OMT_SIZE is 24 so we can create map using array
    // giving us 24x32 bit map info where 0 means there is no
    // collision or ramp and 1 means there is collision or ramp
    unsigned int obstacles_map[24];
    // unsigned int ramps_map[24];

    void Init()
    {
        for (int i = 0; i < 24; i++)
            obstacles_map[i] = 0xFFFFFFFF;
    }

    bool IsObstacle(unsigned int x, unsigned int y)
    {
        cata_assert(x < 24 && y < 24);

        return (obstacles_map[x] | (1 << y));
    }

    // x stand for index in array and y stand for byte in array
    void UpdateObstaclesMap(unsigned int x, unsigned int y, bool set_bit_to_one)
    {
        cata_assert(x < 24 && y < 24);

        if (set_bit_to_one)
            obstacles_map[x] ^= (1 << y);
        else
            obstacles_map[x] &= (1 << y);
    }
};

class vehicle::autodrive_controller
{
private:
    std::array<vehicle_profile, NUM_ORIENTATIONS> vehicle_collision_profiles;
    // std::array<unsigned short, NUM_ORIENTATIONS> all_vehicle_orientations;
    std::vector<navigation_step> current_path;

    const vehicle &driven_vehicle;
    const Character &driver;

    // Overmap terrain data
    tripoint_abs_omt current_omt;
    tripoint_abs_omt next_omt;

    // Vehicle info
    bool vehicle_land = false;
    bool vehicle_water = false;
    bool vehicle_air = false;

    // Max speed for tiles per second for vehicle
    int max_velocity_tps = 0;
    // Set it from max_speed_tps
    int max_velocity = 0;
    int vehicle_acceleration = 0;

    // Obstacle map center around current_omt(3x3 omt map info)
    omt_nav_info nav_infos[3][3];
    tripoint_bub_ms current_nav_goal;

public:
    autodrive_controller(const vehicle &_driven_vehicle, const Character &_driver) : driven_vehicle(_driven_vehicle), driver(_driver) {}

    const Character &GetDriver() const { return driver; }

    omt_nav_info &GetNavInfo(unsigned int x, unsigned int y) { return nav_infos[x][y]; }

    // Called eachtime ComputePath runs
    void UpdateData();

    void ComputeCoordinates();
    void ComputeObstacles();

    // Checks if block at a given area can be driven on
    // Returns true if you can drive on given block
    bool CheckBlockDriveability(const tripoint_bub_ms position);

    // Gets vehicle collision profile at a given angle
    void GetVehicleProfile(vehicle_profile &profile, int vehicle_angle);

    std::vector<navigation_step> &GetCurrentPath() { return current_path; }

    bool IsAreaOccupied(tripoint_abs_ms part_abs_position);
    bool IsVehicleColliding(navigation_step &step, bool check_all);
    void UpdateMoveCost(navigation_step &step);
    // Only computes path to destination
    void PathFinder(std::vector<navigation_step> &path, navigation_step &root_step);

    // Computes and updates given path
    void ComputePath(std::vector<navigation_step> &path);
};

void vehicle::autodrive_controller::UpdateData()
{
    // Overmap terrain navigation data
    current_omt = driven_vehicle.global_omt_location();
    next_omt = driver.omt_path.back();

    // Vehicle info
    vehicle_land = driven_vehicle.valid_wheel_config();
    vehicle_water = driven_vehicle.can_float();
    vehicle_air = driven_vehicle.has_sufficient_rotorlift();

    max_velocity_tps = MAX_SPEED_TPS;
    max_velocity = max_velocity_tps * VMIPH_PER_TPS;

    vehicle_acceleration = driven_vehicle.acceleration(true, 0);

    for (int i = 0; i < NUM_ORIENTATIONS; i++)
        GetVehicleProfile(vehicle_collision_profiles.at(i), i * TURNING_INCREMENT);

    // Navigation data
    ComputeCoordinates();
    ComputeObstacles();
}

void vehicle::autodrive_controller::ComputeCoordinates()
{
    const map &here = get_map();

    int omt_chunk_x = next_omt.x() * OMT_SIZE + OMT_SIZE / 2 - 1;
    int omt_chunk_y = next_omt.y() * OMT_SIZE + OMT_SIZE / 2 - 1;

    tripoint_abs_ms position_abs{omt_chunk_x, omt_chunk_y, next_omt.z()};
    current_nav_goal = here.bub_from_abs(position_abs);

    /*if (current_omt != next_omt)
    {
    }
    else // Just set goal to center to current omt
    {
    }*/
}

void vehicle::autodrive_controller::ComputeObstacles()
{
    const map &here = get_map();

    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            tripoint_abs_omt omt{current_omt.x() + (i - 1), current_omt.y() + (j - 1), current_omt.z()};

            // I don't think you can acces omt as a class so we have to do this shit
            int omt_chunk_size_x = omt.x() * OMT_SIZE;
            int omt_chunk_size_y = omt.y() * OMT_SIZE;

            omt_nav_info &nav_info = GetNavInfo(i, j);
            nav_info.Init();

            for (int x = omt_chunk_size_x; x < omt_chunk_size_x + OMT_SIZE; x++)
            {
                for (int y = omt_chunk_size_y; y < omt_chunk_size_y + OMT_SIZE; y++)
                {
                    tripoint_abs_ms block_position_abs{x, y, omt.z()};
                    tripoint_bub_ms block_position = here.bub_from_abs(block_position_abs);

                    if (here.inbounds(block_position) && CheckBlockDriveability(block_position))
                        nav_info.UpdateObstaclesMap(x - omt_chunk_size_x, y - omt_chunk_size_y, false);
                }
            }

            for (int b = 0; b < 24; b++)
                printf("%u\n", nav_info.obstacles_map[b]);
        }
    }
}

bool vehicle::autodrive_controller::CheckBlockDriveability(const tripoint_bub_ms position)
{
    const map &here = get_map();

    // Check if another vehicle is there; tiles occupied by the current vehicle are evidently drivable
    const optional_vpart_position ovp = here.veh_at(position);

    if (ovp)
    {
        // Known corner case: some furniture can be driven over, but will collide with
        // wheel parts; if the vehicle starts over such a furniture we'll mark that tile
        // as safe and may collide with it by turning; however if we mark it unsafe
        // we'll have no viable paths away from the starting point.
        return &ovp->vehicle() == &driven_vehicle;
    }

    const tripoint_abs_ms position_abs = here.getglobal(position);

    // driver must see the tile or have seen it before in order to plan a route over it
    if (!driver.sees(position))
    {
        if (!driver.is_avatar())
            return false;

        const avatar &avatar = *driver.as_avatar();
        if (!avatar.is_map_memory_valid())
        {
            debugmsg("autodrive querying uninitialized map memory at %s", position_abs.to_string());
            return false;
        }

        if (avatar.get_memorized_tile(position_abs) == mm_submap::default_tile)
        {
            // apparently open air doesn't get memorized, so pretend it is or else
            // we can't fly helicopters due to the many unseen tiles behind the driver
            if (!(vehicle_air && here.ter(position) == ter_t_open_air))
                return false;
        }
    }

    // Check for creatures
    Creature *critter = get_creature_tracker().creature_at(position, true);
    if (critter && driver.sees(*critter))
        return false;

    // Don't drive over visible traps
    if (here.can_see_trap_at(position.raw(), driver))
        return false;

    // Check for furniture that hinders movement; furniture with 0 move cost can be driven on
    const furn_id furniture = here.furn(position);
    if (furniture != furn_str_id::NULL_ID() && furniture.obj().movecost != 0)
        return false;

    // Check if terrain isn't void, cause void has collision in this game
    const ter_id terrain = here.ter(position);
    if (terrain == ter_str_id::NULL_ID())
        return false;

    // Open air is an obstacle to non-flying vehicles; it is drivable for flying vehicles
    if (terrain == ter_t_open_air)
        return vehicle_air;

    const ter_t &terrain_type = terrain.obj();
    // Watercraft can drive on water
    if (vehicle_water && terrain_type.has_flag(ter_furn_flag::TFLAG_SWIMMABLE))
        return true;

    // Remaining checks are for land-based navigation
    if (!vehicle_land)
        return false;

    // NOLINTNEXTLINE(bugprone-branch-clone)
    if (terrain_type.movecost <= 0)
    {
        // walls and other impassable terrain
        return false;
    }
    else if (terrain_type.movecost == 2 || terrain_type.has_flag(ter_furn_flag::TFLAG_NOCOLLIDE))
    {
        // terrain with neutral move cost or tagged with NOCOLLIDE will never cause
        // collisions
        return true;
    }
    else if (terrain_type.bash.str_max >= 0 && !terrain_type.bash.bash_below)
    {
        // bashable terrain (but not bashable floors) will cause collisions
        return false;
    }
    else if (terrain_type.has_flag(ter_furn_flag::TFLAG_LIQUID))
    {
        // water and lava
        return false;
    }

    return true;
}

// Takes integer number and makes either -1, 0, 1
static int NumberToOne(int num)
{
    return ((num == 0) ? 0 : ((num > 0) ? 1 : -1));
}

void vehicle::autodrive_controller::GetVehicleProfile(vehicle_profile &profile, int vehicle_angle)
{
    profile.Clear();

    // Used for translating part position around pivot
    tileray &ray_direction = profile.ray_direction;
    ray_direction.init(units::from_degrees(vehicle_angle));

    // This is center of vehicle, that is used to rotate
    // all the parts around it in car when it moves
    const point pivot = driven_vehicle.pivot_point();

    for (const vehicle_part &part : driven_vehicle.parts)
    {
        // This should never happen
        if (part.removed)
            continue;

        tripoint pos;
        driven_vehicle.coord_translate(ray_direction, pivot, part.mount, pos);
        profile.AddOccupiedArea({pos.x, pos.y});
    }

    // Aren't rotors one tile higher than frame and other vehicle parts
    /*for (int part_num : driven_vehicle.rotors)
    {
        const vehicle_part &part = driven_vehicle.part(part_num);
        const int diameter = part.info().rotor_info->rotor_diameter;
        const int radius = (diameter + 1) / 2;

        if (radius > 0)
        {
            tripoint pos;
            driven_vehicle.coord_translate(tdir, pivot, part.mount, pos);
            for (tripoint rotor_pos : points_in_radius(pos, radius))
                profile.AddOccupiedArea({rotor_pos.x, rotor_pos.y});
        }
    }*/

    std::unordered_set<point> &occupied_areas = profile.occupied_areas;

    // When moving at 45 degree angle I think collision might work differently
    // Edit: It work differently, I don't know how for now
    /*if (vehicle_angle % 90 == 45)
    {
    }*/

    // Check if vehicle will be moving at an angle, if so we have an tough nut to crack
    if (vehicle_angle % 90 != 0)
    {
        // Advance direction by 5 tiles, reason is the way vehicle works. Moving by 5
        // tiles should always move vehicle in x and y whenever it's at a degree
        // Make it 100 just to be safe
        ray_direction.advance(100);

        int x_movement = NumberToOne(ray_direction.dx());
        int y_movement = NumberToOne(ray_direction.dy());

        std::unordered_set<point> ray_areas_set{0};

        // Checks all possible places we could move to when moving
        for (int i = 0; i <= 2; i++)
        {
            point increment;
            if (i == 0)
                increment = point(x_movement, 0);
            else if (i == 1)
                increment = point(0, y_movement);
            else
                increment = point(x_movement, y_movement);

            for (const point &occupied_area : occupied_areas)
            {
                point ray_area = point(occupied_area.x, occupied_area.y);
                ray_area += increment;

                point ray_pos{ray_area.x, ray_area.y};

                if (!profile.IsAreaOccupied(ray_pos) && ray_areas_set.find(ray_pos) == ray_areas_set.end())
                    ray_areas_set.emplace(ray_pos);
            }
        }

        for (const auto &ray_area : ray_areas_set)
            profile.AddRayArea(point(ray_area.x, ray_area.y));
    }
    else
    {
        // Is looking directly in either north, sout, east, west so
        // when moving by all the parts will move in either x or y
        ray_direction.advance(1);
        const point increment(ray_direction.dx(), ray_direction.dy());

        for (const point &occupied_area : occupied_areas)
        {
            point ray_area = point(occupied_area.x, occupied_area.y);
            ray_area += increment;

            if (!profile.IsAreaOccupied({ray_area.x, ray_area.y}))
                profile.AddRayArea(ray_area);
        }
    }
}

// Returns true if we can't check for omt info inside nav_info
static bool IsNavInfoOutOfBounds(tripoint omt)
{
    return abs(omt.x) > 1 || abs(omt.y) > 1;
}

bool vehicle::autodrive_controller::IsAreaOccupied(tripoint_abs_ms part_abs_position)
{
    tripoint_abs_omt part_abs_omt{part_abs_position.raw()};
    ms_to_omt(part_abs_omt.raw());

    tripoint omt_difference = part_abs_omt.raw() - current_omt.raw();

    if (IsNavInfoOutOfBounds(omt_difference))
        return true;

    omt_nav_info &nav_info = GetNavInfo(omt_difference.x, omt_difference.y);

    tripoint part_module_position(part_abs_position.x() % OMT_SIZE, part_abs_position.y() % OMT_SIZE, part_abs_position.z());

    if (nav_info.IsObstacle(part_module_position.x, part_module_position.y))
        return true;

    return false;
}

bool vehicle::autodrive_controller::IsVehicleColliding(navigation_step &step, bool check_all)
{
    if (step.angle < 0 || step.angle > 345)
    {
        // printf("Invalid Angle: %i\n", step.angle);
        return true;
    }

    // cata_assert(step.angle >= 0 || step.angle <= 345);

    vehicle_profile &profile = vehicle_collision_profiles[step.angle / TURNING_INCREMENT];

    if (check_all)
    {
        for (point occupied_area : profile.occupied_areas)
        {
            tripoint_rel_ms occupied_area_tripoint = tripoint_rel_ms(occupied_area.x, occupied_area.y, 0);
            tripoint_abs_ms part_abs_position = step.position_abs + occupied_area_tripoint;

            if (IsAreaOccupied(part_abs_position))
                return true;
        }
    }
    else
    {
        for (point ray_area : profile.ray_areas)
        {
            tripoint_rel_ms occupied_area_tripoint = tripoint_rel_ms(ray_area.x, ray_area.y, 0);
            tripoint_abs_ms part_abs_position = step.position_abs + occupied_area_tripoint;

            if (IsAreaOccupied(part_abs_position))
                return true;
        }
    }

    return false;
}

static void ClampAngle(int &angle)
{
    while (angle > 345)
        angle -= 345;
    while (angle < 0)
        angle += 345;
}

static void BuildPath(std::vector<navigation_step> &path, navigation_step &last_step)
{
    printf("Building path\n");

    navigation_step *previous_step = &last_step;

    while (previous_step->previous != nullptr)
    {
        path.push_back(*previous_step);
        previous_step = previous_step->previous;
    }
}

void vehicle::autodrive_controller::UpdateMoveCost(navigation_step &step)
{
    step.move_cost = 0;

    // The deeper the longer
    step.move_cost += step.depth;

    // Apply angle difference, looking away from target makes it much worse
    // cause often we can't move fast when turning due to tight spot
    tripoint omt_diff = current_nav_goal.raw() - step.position_abs.raw();

    int goal_angle = int(atan2(omt_diff.x, omt_diff.y) * (180 / M_PI));
    int vehicle_angle = int(units::to_degrees(driven_vehicle.face.dir()));

    step.move_cost += std::abs((goal_angle - vehicle_angle) / TURNING_INCREMENT) * 2;

    // Distance
    step.move_cost += sqrt(omt_diff.x * omt_diff.x + omt_diff.y * omt_diff.y) / 2;
}

void vehicle::autodrive_controller::PathFinder(std::vector<navigation_step> &path, navigation_step &root_step)
{
    path.clear();

    std::vector<navigation_step> navmesh;
    std::priority_queue<navigation_step *, std::vector<navigation_step *>, navigation_step> navmesh_set;

    navmesh.push_back(root_step);
    navmesh_set.push(&navmesh.back());

    int loops_amount = 0;

    while (!navmesh_set.empty() && loops_amount++ < 1000)
    {
        navigation_step *previous_step = navmesh_set.top();
        navmesh_set.pop();

        for (int accelerate = -1; accelerate <= 1; accelerate++)
        {
            // We are stuck at single place
            if (navmesh.back().velocity == 0 && accelerate == 0)
                continue;

            // No more speeding up
            if (accelerate != 0 && std::abs(previous_step->velocity) >= max_velocity)
                continue;

            for (int steer = -1; steer <= 1; steer++)
            {
                // Copy last step to modify it
                navigation_step new_step = *previous_step;

                // Set previous node and increase depth from previous node
                new_step.previous = previous_step;
                new_step.depth++;

                // Compute acceleration
                if (accelerate != 0)
                {
                    if (accelerate == 1)
                        new_step.velocity += vehicle_acceleration;
                    else if (accelerate == -1)
                        new_step.velocity -= vehicle_acceleration * 0.6;

                    // Apply max_velocity, also applies negative value so velocity will have reverstion
                    if (std::abs(new_step.velocity) > max_velocity)
                        new_step.velocity = accelerate * max_velocity;
                }

                // Okay so if our speed was zero and let's say we set our wish speed to 2 tps with 1 tps acceleration.
                // We will get to 1 tps speed in the next second, but we won't move even those if we wait another second
                // and get 2 tps speed we would move 2 tps than. Also works the same way if we start reversing, cause when
                // reversing we have to have speed of 0 for 1 tps
                if (previous_step->velocity != 0)
                    new_step.travelled_distance += new_step.velocity;

                // Here we move and check for collision
                while (new_step.travelled_distance >= VMIPH_PER_TPS)
                {
                    new_step.travelled_distance -= VMIPH_PER_TPS;

                    tileray &move_dir = new_step.vehicle_direction;

                    int prev_x = move_dir.dx();
                    int prev_y = move_dir.dy();

                    move_dir.advance(1);

                    new_step.position_abs += tripoint_rel_ms(move_dir.dx() - prev_x, move_dir.dy() - prev_y, 0);

                    // If at current position we collide with anything
                    // it means, we should discard this future position
                    if (IsVehicleColliding(new_step, false))
                    {
                        break;
                        continue;
                    }
                }

                if (IsVehicleColliding(new_step, true))
                    continue;

                // Adjust steering after moving
                if (steer != 0)
                {
                    new_step.angle += steer * TURNING_INCREMENT;
                    ClampAngle(new_step.angle);
                    new_step.vehicle_direction.init(units::from_degrees(new_step.angle));
                }

                // Update cost after everything is calculated
                UpdateMoveCost(new_step);

                navmesh.push_back(new_step);
                navmesh_set.push(&navmesh.back());

                tripoint_abs_omt part_abs_omt{new_step.position_abs.raw()};
                ms_to_omt(part_abs_omt.raw());

                printf("next_omt x: %i, y: %i\n", next_omt.x(), next_omt.y());
                printf("part_abs_omt x: %i, y: %i\n", part_abs_omt.x(), part_abs_omt.y());

                if (part_abs_omt.raw() == next_omt.raw())
                    return BuildPath(path, new_step);
            }
        }
    }

    printf("Navmesh size: %lu\n", navmesh.size());
}

void vehicle::autodrive_controller::ComputePath(std::vector<navigation_step> &path)
{
    UpdateData();

    const tripoint_abs_ms vehicle_position_abs = driven_vehicle.global_square_location();

    // Is following path, that we previously calculated
    bool following_path = false;

    if (!path.empty())
    {
        // Compare previous path to check if we need to recalculate
        if (path.size() >= 2)
        {
            // navigation_step last_step = path.back();
            navigation_step last_last_step = path.at(path.size() - 2);

            // Check if we correctly followed path, if not clear it
            // If predicted position is the same as current one
            // If speed is the same we target one
            // And if the direction is the same as target one
            if (last_last_step.position_abs == vehicle_position_abs && last_last_step.velocity == driven_vehicle.velocity && last_last_step.vehicle_direction.dir() == driven_vehicle.move.dir())
                following_path = true;
            else
                path.clear();
        }

        // Remove last node
        // if (!path.empty())
        //     path.pop_back();
    }

    // Following good path, keep going
    if (!path.empty() && following_path)
        return;

    navigation_step root_step;

    // Positional info
    root_step.position_abs = vehicle_position_abs;
    root_step.vehicle_direction = driven_vehicle.face;
    root_step.angle = int(units::to_degrees(driven_vehicle.face.dir()));

    // Velocity releated info
    // root_step.travelled_distance = driven_vehicle.
    root_step.cruise_velocity = driven_vehicle.cruise_velocity;
    root_step.velocity = driven_vehicle.velocity;

    // Pathfinder
    root_step.previous = nullptr;
    root_step.move_cost = 0;
    root_step.depth = 0;

    PathFinder(path, root_step);
}

std::vector<std::tuple<point, int, std::string>> vehicle::get_debug_overlay_data() const
{
    std::vector<std::tuple<point, int, std::string>> ret;

    return ret;
}

autodrive_result vehicle::do_autodrive(Character &driver)
{
    if (!is_autodriving)
        return autodrive_result::abort;

    if (!player_in_control(driver) || skidding)
    {
        driver.add_msg_if_player(m_warning, _("You lose control as the vehicle starts skidding."));
        stop_autodriving(false);
        return autodrive_result::abort;
    }

    const tripoint_abs_ms veh_pos = global_square_location();
    const tripoint_abs_omt veh_omt = project_to<coords::omt>(veh_pos);
    std::vector<tripoint_abs_omt> &omt_path = driver.omt_path;

    while (!omt_path.empty() && veh_omt.xy() == omt_path.back().xy())
        omt_path.pop_back();

    if (omt_path.empty())
    {
        stop_autodriving(false);
        return autodrive_result::finished;
    }

    if (!active_autodrive_controller)
        active_autodrive_controller = std::make_shared<autodrive_controller>(*this, driver);

    if (&active_autodrive_controller->GetDriver() != &driver)
    {
        debugmsg("Driver changed while auto-driving");
        stop_autodriving();
        return autodrive_result::abort;
    }

    std::vector<navigation_step> &current_path = active_autodrive_controller->GetCurrentPath();
    active_autodrive_controller->ComputePath(current_path);

    if (current_path.empty())
    {
        // message handles pathfinding failure either due to obstacles or inability to see
        driver.add_msg_if_player(_("Can't see a path forward."));
        stop_autodriving(false);
        return autodrive_result::abort;
    }

    navigation_step &next_step = current_path.back();

    if (next_step.position_abs.xy() != veh_pos.xy())
    {
        debugmsg("compute_next_step returned an invalid result");
        stop_autodriving();
        return autodrive_result::abort;
    }

    current_path.pop_back();

    cruise_velocity = next_step.velocity;

    int goal_angle = next_step.angle;
    int vehicle_angle = int(units::to_degrees(face.dir()));

    pldrive(driver, {NumberToOne(goal_angle - vehicle_angle), 0});

    return autodrive_result::ok;
}

void vehicle::stop_autodriving(bool apply_brakes)
{
    if (!is_autodriving && !is_patrolling && !is_following)
        return;

    if (apply_brakes)
        cruise_velocity = 0;

    is_autodriving = false;
    is_patrolling = false;
    is_following = false;
    autopilot_on = false;
    autodrive_local_target = tripoint_zero;
    collision_check_points.clear();
    active_autodrive_controller.reset();
}
